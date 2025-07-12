from dotenv import load_dotenv
load_dotenv()

import os
import logging
import torch
from torch.amp import autocast
from datasets import load_dataset
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException
from langchain_huggingface import HuggingFaceEmbeddings
import argparse
import re

from tenacity import retry, wait_exponential, stop_after_attempt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser("Index MedMCQA into Pinecone")
parser.add_argument("--data_percent", type=float, default=float(os.getenv("DATA_PERCENT","0.1")),
                    help="Fraction of dataset to index (0.0–1.0)")
parser.add_argument("--batch_size", type=int, default=int(os.getenv("INDEX_BATCH_SIZE","32")),
                    help="Embedding/upsert batch size")
args = parser.parse_args()

data_percent = max(0.0, min(1.0, args.data_percent))
BATCH_SIZE   = args.batch_size
INDEX_NAME   = os.getenv("PINECONE_INDEX","medicalbot")
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL","BAAI/bge-large-en-v1.5")

logger.info(f"Indexing {data_percent*100:.2f}% of the dataset (batch_size={BATCH_SIZE})")

# Check PyTorch version, since we use autocast
if torch.__version__ < "2.6.0":
    raise RuntimeError("PyTorch ≥2.6.0 required")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Embedding on device: {device}")

api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise RuntimeError("Missing PINECONE_API_KEY in .env")
pc = Pinecone(api_key=api_key, environment=os.getenv("PINECONE_ENVIRONMENT","us-east1-gcp"))

# Create or clear index
if INDEX_NAME not in pc.list_indexes().names():
    logger.info(f"Creating index '{INDEX_NAME}' (dim=1024)")
    pc.create_index(name=INDEX_NAME, dimension=1024, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
else:
    logger.info(f"Clearing existing index '{INDEX_NAME}'")
    idx = pc.Index(INDEX_NAME)
    try:
        idx.delete(delete_all=True, namespace="")
    except NotFoundException:
        logger.info("Index was already empty")

# Load & slice dataset
logger.info("Loading MedMCQA train split…")
ds = load_dataset("openlifescienceai/medmcqa", split="train")
total = len(ds)
cutoff = int(total * data_percent)
if data_percent < 1.0:
    ds = ds.select(range(cutoff))
    logger.info(f"Selected {len(ds)} / {total} examples ({data_percent*100:.2f}%)")
else:
    logger.info(f"Using all {total} examples")

# Prepare text chunks & metadata
ids, texts, metas = [], [], []
opt_map = {0:"A",1:"B",2:"C",3:"D"}
opt_rev = {"a":0,"b":1,"c":2,"d":3}

for item in tqdm(ds, desc="Chunking"):
    qid   = str(item["id"])
    # ensure strings
    q     = str(item.get("question",""))
    exp   = str(item.get("exp",""))
    opts  = {
        "A": str(item.get("opa","")),
        "B": str(item.get("opb","")),
        "C": str(item.get("opc","")),
        "D": str(item.get("opd","")),
    }
    ctype = str(item.get("choice_type","single"))

    # parse numbered sub-options
    sub = {
        n: txt.strip()
        for n,txt in re.findall(r"(\d+)\.\s*(.*?)(?=\n\d+\.|$)", q, re.S)
    }

    # extract from explanation if present
    m    = re.search(r"Ans\. is\s*['\"]([A-D](?:\s*,\s*[A-D])*)['\"]", exp, re.I)
    corr = [opt_rev[c.lower()] for c in m.group(1).split(",")] if m else []

    # fallback to cop field
    cop = item.get("cop", 0)
    if isinstance(cop,(int,float)):
        cop = int(cop)
    elif isinstance(cop,list):
        cop = [int(x) for x in cop]
    else:
        cop = 0

    # determine correct indices
    if corr and all(0 <= c < 4 for c in corr):
        correct = corr
    else:
        if ctype=="multi" and isinstance(cop,list):
            correct = [c for c in cop if 0 <= c < 4]
        else:
            correct = [cop] if 0 <= cop < 4 else [0]

    # handle "all of the above" and also the sub-options
    if ctype=="multi" and (
        opts["D"].lower().startswith("all") or len(correct)>1
    ):
        if sub:
            correct  = [int(k)-1 for k in sub]
            answers  = list(sub.values())
        else:
            answers  = [opts[opt_map[c]] for c in correct]
    else:
        answers = [opts[opt_map[c]] for c in correct]

    # Chunking text
    block = [f"Question: {q}"] + [f"{L}: {O}" for L,O in opts.items()]
    block += [
        f"Correct: {', '.join(answers) or 'Unknown'}",
        f"Explanation: {exp}"
    ]
    text = "\n".join(block)

    ids.append(qid)
    texts.append(text)
    md = {
        "text": text,
        "correct_options": [str(c) for c in correct],
        "correct_answer_text": ", ".join(answers),
        "subject": str(item.get("subject_name","")),
        "topic":   str(item.get("topic_name","")),
        "choice_type": ctype,
    }
    for i,(n,v) in enumerate(sub.items(),1):
        md[f"sub_option_{i}"] = v
    metas.append(md)

logger.info(f"Prepared {len(texts)} text chunks")

#  Embedder setup
#TODO: Probably check medbase embedding model here(BioBERT, ClinicalBert), and also OpenAI embedding model (would potentially reduce the size of docker image)
embedder = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device":device},
    encode_kwargs={"normalize_embeddings":True},
)

# Create embeddings
vectors = []
logger.info("Generating embeddings…")
for i in tqdm(range(0,len(texts),BATCH_SIZE), desc="Embedding"):
    with autocast(device):
        vectors.extend(embedder.embed_documents(texts[i:i+BATCH_SIZE]))
logger.info(f"Generated {len(vectors)} embeddings")

# Upsert with retry
index = pc.Index(INDEX_NAME)

@retry(wait=wait_exponential(min=1,max=5), stop=stop_after_attempt(5))
def safe_upsert(batch):
    index.upsert(vectors=batch, namespace="")

logger.info("Upserting vectors…")
for i in tqdm(range(0,len(vectors),BATCH_SIZE), desc="Upsert"):
    batch = [
        (ids[j], vectors[j], metas[j])
        for j in range(i, min(i+BATCH_SIZE, len(vectors)))
    ]
    try:
        safe_upsert(batch)
    except Exception as e:
        logger.error(f"Upsert failed for batch at idx {i}: {e}")

logger.info("Indexing complete!")
