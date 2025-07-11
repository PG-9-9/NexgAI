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
logger = logging.getLogger()

if torch.__version__ < "2.6.0":
    raise RuntimeError("PyTorch ≥2.6.0 required")

# CLI arg: percent of dataset
parser = argparse.ArgumentParser("Index MedMCQA")
parser.add_argument("--data_percent", type=float, default=0.1)
args = parser.parse_args()
p = max(0.0, min(1.0, args.data_percent))
logger.info(f"Indexing {p*100:.1f}% of MedMCQA")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Embedding on {device}")

api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise RuntimeError("Missing PINECONE_API_KEY")
pc = Pinecone(api_key=api_key, environment=os.getenv("PINECONE_ENVIRONMENT","us-east1-gcp"))
INDEX = "medicalbot"

if INDEX not in pc.list_indexes().names():
    pc.create_index(INDEX, dimension=1024, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
else:
    idx = pc.Index(INDEX)
    try:
        idx.delete(delete_all=True, namespace="")
    except NotFoundException:
        pass

ds = load_dataset("openlifescienceai/medmcqa", split="train")
if p < 1.0:
    ds = ds.select(range(int(len(ds)*p)))
logger.info(f"Preparing {len(ds)} examples")

ids, texts, meta = [], [], []
opt_map = {0:"A",1:"B",2:"C",3:"D"}
opt_rev = {"a":0,"b":1,"c":2,"d":3}

for itm in tqdm(ds, desc="Chunking"):
    qid = str(itm["id"])
    q = itm.get("question") or ""
    opts = {L: itm.get(f"op{L.lower()}", "") for L in ("A","B","C","D")}
    exp = itm.get("exp") or ""
    ctype = itm.get("choice_type") or "single"

    sub = {n:txt.strip() for n,txt in
           re.findall(r"(\d+)\.\s*(.*?)(?=\n\d+\.|$)", q, re.S)}

    m = re.search(r"Ans\. is\s*['\"]([A-D](?:\s*,\s*[A-D])*)['\"]", exp, re.I)
    corr = [opt_rev[c.lower()] for c in m.group(1).split(",")] if m else []
    cop = itm.get("cop",0)
    if isinstance(cop,(int,float)): cop=int(cop)
    elif isinstance(cop,list): cop=[int(x) for x in cop]
    else: cop=0

    if corr and all(c in range(4) for c in corr):
        correct = corr
    else:
        correct = corr if (ctype=="multi" and isinstance(cop,list)) else ([cop] if cop in range(4) else [0])

    if ctype=="multi" and (opts["D"].lower().startswith("all") or len(correct)>1):
        if sub:
            correct = [int(k)-1 for k in sub]
            answers = list(sub.values())
        else:
            answers = [opts[opt_map[c]] for c in correct]
    else:
        answers = [opts[opt_map[c]] for c in correct]

    block = [f"Question: {q}"] + [f"{L}: {O}" for L,O in opts.items()]
    block += [f"Correct: {', '.join(answers) or 'Unknown'}",
              f"Explanation: {exp}"]
    txt = "\n".join(block)

    ids.append(qid)
    texts.append(txt)
    md = {
        "text": txt,
        "correct_options": [str(c) for c in correct],
        "correct_answer_text": ", ".join(answers),
        "subject": itm.get("subject_name") or "",
        "topic": itm.get("topic_name") or "",
        "choice_type": ctype,
    }
    for i,(n,v) in enumerate(sub.items(),1):
        md[f"sub_option_{i}"] = v
    meta.append(md)

logger.info("Initializing BGE-large embedder")
embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device":device},
    encode_kwargs={"normalize_embeddings":True},
)

batch = 64
vecs = []
logger.info("Embedding in batches")
for i in tqdm(range(0,len(texts),batch), desc="Embed"):
    with autocast(device):
        vecs.extend(embedder.embed_documents(texts[i:i+batch]))

idx = pc.Index(INDEX)
logger.info("Upserting in batches")
for i in tqdm(range(0,len(vecs),batch), desc="Upsert"):
    chunk = [(ids[j], vecs[j], meta[j]) for j in range(i,min(i+batch,len(vecs)))]
    idx.upsert(vectors=chunk, namespace="")

logger.info("Done indexing.")
