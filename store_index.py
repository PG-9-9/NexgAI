# store_index.py

from dotenv import load_dotenv
load_dotenv()

import os
import logging
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
if device == "cpu":
    logger.warning("GPU not available, falling back to CPU. This will be slower.")

# Load and validate Pinecone credentials
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise RuntimeError("PINECONE_API_KEY not found in .env")

# Initialize Pinecone client and index
pc = Pinecone(api_key=api_key)
index_name = "medicalbot"

existing = pc.list_indexes().names()
if index_name not in existing:
    logger.info(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    logger.info(f"Clearing existing index '{index_name}'...")
    idx = pc.Index(index_name)
    try:
        idx.delete(delete_all=True, namespace="")
    except NotFoundException:
        logger.info("Index was already empty.")

# Load the MedMCQA dataset
logger.info("Loading MedMCQA dataset (train split)...")
try:
    dataset = load_dataset("openlifescienceai/medmcqa", split="train")
    total = len(dataset)
    logger.info(f"Loaded {total} examples.")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise

# Prepare text chunks
ids = []
texts = []
for item in dataset:
    qid = str(item["id"])
    question = item["question"]
    opts = {
        "A": item["opa"],
        "B": item["opb"],
        "C": item["opc"],
        "D": item["opd"],
    }
    text = [f"Question: {question}"]
    for label, opt in opts.items():
        text.append(f"{label}: {opt}")
    ids.append(qid)
    texts.append("\n".join(text))

logger.info(f"Prepared {len(texts)} text chunks for embedding.")

# Initialize embedder with GPU support
try:
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
except Exception as e:
    logger.error(f"Failed to initialize embedder: {e}")
    raise

batch_size = 64  # Adjusted for GPU memory constraints
vectors = []

logger.info("Generating embeddings in batches...")
for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
    batch_texts = texts[i : i + batch_size]
    try:
        batch_vecs = embedder.embed_documents(batch_texts)
        vectors.extend(batch_vecs)
    except Exception as e:
        logger.error(f"Error embedding batch {i // batch_size}: {e}")
        continue
logger.info(f"Embedding complete. Generated {len(vectors)} embeddings.")

# Upsert in batches
index = pc.Index(index_name)
logger.info("Upserting vectors to Pinecone in batches...")
for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting"):
    to_up = [
        (ids[j], vectors[j], {"text": texts[j]})
        for j in range(i, min(i + batch_size, len(vectors)))
    ]
    try:
        index.upsert(vectors=to_up, namespace="")
    except Exception as e:
        logger.error(f"Error upserting batch {i // batch_size}: {e}")
        continue
logger.info(f"Upsert complete. Indexed {len(vectors)} vectors.")