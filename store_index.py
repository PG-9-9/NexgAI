# store_index.py

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# require PyTorch ≥2.6.0
if torch.__version__ < "2.6.0":
    raise RuntimeError(f"PyTorch {torch.__version__} too old; upgrade to ≥2.6.0")

# CLI: % of dataset to index
parser = argparse.ArgumentParser("Index MedMCQA into Pinecone")
parser.add_argument("--data_percent", type=float, default=0.1,
                    help="Portion of dataset (0.0–1.0) to process")
args = parser.parse_args()
data_percent = max(0.0, min(1.0, args.data_percent))
logger.info(f"Indexing {data_percent*100:.1f}% of MedMCQA")

# choose GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
if device == "cpu":
    logger.warning("No GPU detected; embedding will be slower")

# init Pinecone
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise RuntimeError("Missing PINECONE_API_KEY in .env")
pc = Pinecone(api_key=api_key, environment=os.getenv("PINECONE_ENVIRONMENT","us-east1-gcp"))
index_name = "medicalbot"

# create or clear index (now dim=1024 for BGE-large)
if index_name not in pc.list_indexes().names():
    logger.info(f"Creating Pinecone index '{index_name}' (dim=1024)")
    pc.create_index(name=index_name, dimension=1024, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
else:
    logger.info(f"Clearing existing index '{index_name}'")
    idx = pc.Index(index_name)
    try:
        idx.delete(delete_all=True, namespace="")
    except NotFoundException:
        logger.info("Index was already empty")

# load MedMCQA
logger.info("Loading MedMCQA (train split)...")
ds = load_dataset("openlifescienceai/medmcqa", split="train")
total = len(ds)
logger.info(f"Total examples: {total}")
if data_percent < 1.0:
    cutoff = int(total * data_percent)
    ds = ds.select(range(cutoff))
    logger.info(f"Selected {len(ds)} examples ({data_percent*100:.1f}%)")

# prepare chunks & metadata
ids, texts, metadatas = [], [], []
option_map = {0:"A",1:"B",2:"C",3:"D"}
option_map_rev = {"a":0,"b":1,"c":2,"d":3}
valid = set(option_map.values())

for item in ds:
    qid = str(item["id"])
    question = item.get("question") or "Unknown"
    opts = {
        "A": item.get("opa") or "Unknown",
        "B": item.get("opb") or "Unknown",
        "C": item.get("opc") or "Unknown",
        "D": item.get("opd") or "Unknown",
    }
    explanation = item.get("exp") or "No explanation provided."
    choice_type = item.get("choice_type") or "single"

    # parse numbered sub-options in question
    sub_options = {n: txt.strip() for n,txt in
        re.findall(r"(\d+)\.\s*(.*?)(?=\n\d+\.|$)", question, re.S)
    }

    # try extract correct from explanation
    corr_exp = []
    match = re.search(r"Ans\. is\s*['\"]([A-D](?:\s*,\s*[A-D])*)['\"]", explanation, re.I)
    if match:
        corr_exp = [option_map_rev[c.lower()] for c in match.group(1).split(",")]

    # handle cop field fallback
    cop = item.get("cop",0)
    if isinstance(cop,(int,float)):
        cop = int(cop)
    elif isinstance(cop,list):
        cop = [int(x) for x in cop]
    else:
        cop = 0

    # determine correct options
    if corr_exp and all(c in range(4) for c in corr_exp):
        correct = corr_exp
    else:
        if choice_type=="multi" and isinstance(cop,list):
            correct = [c for c in cop if c in range(4)]
        else:
            correct = [cop] if cop in range(4) else [0]

    # expand “all of the above” or sub-options
    if choice_type=="multi" and (opts["D"].lower().startswith("all") or len(correct)>1):
        if sub_options:
            correct = [int(k)-1 for k in sub_options]
            answers = list(sub_options.values())
        else:
            answers = [opts[option_map[c]] for c in correct]
    else:
        answers = [opts[option_map[c]] for c in correct]

    # build text block
    block = [f"Question: {question}"]
    for L,O in opts.items():
        block.append(f"{L}: {O}")
    block.append(f"Correct: {', '.join(answers) or 'Unknown'}")
    block.append(f"Explanation: {explanation}")
    text = "\n".join(block)

    ids.append(qid)
    texts.append(text)
    md = {
        "text": text,
        "correct_options":[str(c) for c in correct],
        "correct_answer_text": ", ".join(answers),
        "subject": item.get("subject_name") or "Unknown",
        "topic": item.get("topic_name") or "Unknown",
        "choice_type": choice_type,
    }
    for i,(n,so) in enumerate(sub_options.items(),1):
        md[f"sub_option_{i}"] = so
    metadatas.append(md)

logger.info(f"Prepared {len(texts)} chunks")

# init BGE-large embedder
embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device":device},
    encode_kwargs={"normalize_embeddings":True},
)

batch_size = 64  
vectors = []
logger.info("Embedding in batches with mixed precision...")
for i in tqdm(range(0, len(texts), batch_size), desc="Embed"):
    batch = texts[i:i+batch_size]
    with autocast(device):
        vecs = embedder.embed_documents(batch)
    vectors.extend(vecs)
logger.info(f"Generated {len(vectors)} vectors")

# upsert in batches
index = pc.Index(index_name)
logger.info("Upserting vectors to Pinecone...")
for i in tqdm(range(0, len(vectors), batch_size), desc="Upsert"):
    chunk = [
        (ids[j], vectors[j], metadatas[j])
        for j in range(i, min(i+batch_size, len(vectors)))
    ]
    index.upsert(vectors=chunk, namespace="")
logger.info("Indexing complete.")
