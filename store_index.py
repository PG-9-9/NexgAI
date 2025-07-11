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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Verify PyTorch version
required_torch_version = "2.6.0"
if torch.__version__ < required_torch_version:
    raise RuntimeError(f"PyTorch version {torch.__version__} is too old. Please upgrade to {required_torch_version} or higher.")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Index MedMCQA dataset into Pinecone.")
parser.add_argument(
    "--data_percent",
    type=float,
    default=0.1,
    help="Percentage of dataset to process (0.0 to 1.0)",
)
args = parser.parse_args()
data_percent = max(0.0, min(1.0, args.data_percent))  # Clamp between 0 and 1
logger.info(f"Using {data_percent*100}% of the dataset.")

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
        dimension=768,  # For bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12
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
    if data_percent < 1.0:
        dataset = dataset.select(range(int(total * data_percent)))
        logger.info(f"Selected {len(dataset)} examples ({data_percent*100}%).")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise

# Prepare text chunks and metadata
ids = []
texts = []
metadatas = []
option_map = {0: "A", 1: "B", 2: "C", 3: "D"}  # Zero-based mapping
option_map_reverse = {"a": 0, "b": 1, "c": 2, "d": 3}  # For parsing explanation
valid_cops = set(option_map.keys())

for item in dataset:
    qid = str(item["id"])
    question = item["question"] or "Unknown"
    opts = {
        "A": item["opa"] or "Unknown",
        "B": item["opb"] or "Unknown",
        "C": item["opc"] or "Unknown",
        "D": item["opd"] or "Unknown",
    }
    explanation = item["exp"] if item["exp"] else "No explanation provided."
    choice_type = item["choice_type"] if item["choice_type"] is not None else "single"

    # Parse numbered sub-options from question (e.g., "1. Venous ulcer 2. Pulmonary embolism")
    sub_options = {}
    sub_option_matches = re.findall(r"(\d+)\.\s*(.*?)(?=\n\d+\.|$)", question, re.S)
    for num, text in sub_option_matches:
        sub_options[str(int(num))] = text.strip()

    # Parse explanation for correct option(s)
    correct_options_from_exp = []
    if explanation != "No explanation provided.":
        matches = re.findall(r"Ans\. is\s*['\"]([a-d](?:\s*,\s*[a-d])*)['\"]", explanation, re.IGNORECASE)
        if matches:
            options = matches[0].replace(" ", "").split(",")
            correct_options_from_exp = [option_map_reverse.get(opt.lower()) for opt in options if opt.lower() in option_map_reverse]

    # Handle cop
    cop = item["cop"] if item["cop"] is not None else 0
    if isinstance(cop, (int, float)):
        cop = int(cop)
    elif isinstance(cop, list):
        cop = [int(c) for c in cop]
    else:
        cop = 0
        logger.warning(f"Invalid cop type {type(item['cop'])} for ID {qid}. Defaulting to 0.")

    # Determine correct options
    if correct_options_from_exp and all(c in valid_cops for c in correct_options_from_exp):
        correct_options = correct_options_from_exp
    else:
        if choice_type == "multi" and isinstance(cop, list):
            correct_options = [c for c in cop if c in valid_cops]
        else:
            correct_options = [cop] if cop in valid_cops else [0]

    # Handle "All are true" or multi-choice with sub-options
    if choice_type == "multi" and (opts["D"].lower() == "all are true" or len(correct_options) > 1):
        if sub_options:
            correct_options = [int(k) - 1 for k in sub_options.keys()]  # Map to all sub-options
            correct_answers = list(sub_options.values())
        else:
            # Parse sub-options from options (e.g., "1,2,3 & 4")
            sub_option_nums = []
            for opt in correct_options:
                if opts[option_map[opt]].lower() != "all are true":
                    nums = re.findall(r"\d+", opts[option_map[opt]])
                    sub_option_nums.extend([int(n) - 1 for n in nums if n in sub_options])
            correct_options = list(set(sub_option_nums)) if sub_option_nums else correct_options
            correct_answers = [sub_options[str(c + 1)] for c in correct_options if str(c + 1) in sub_options]
    else:
        correct_answers = [opts[option_map[c]] for c in correct_options]

    # Log discrepancies or invalid cop
    if isinstance(cop, (int, float)) and cop not in valid_cops:
        logger.warning(f"Invalid cop={cop} for ID {qid}. Using {correct_options} from explanation or default.")
    elif isinstance(cop, list) and any(c not in valid_cops for c in cop):
        logger.warning(f"Invalid cop values in {cop} for ID {qid}. Using {correct_options} from explanation or default.")
    elif correct_options_from_exp and set(correct_options_from_exp) != set(correct_options):
        logger.warning(f"Discrepancy for ID {qid}: cop={cop} ({[option_map[c] for c in ([cop] if isinstance(cop, int) else cop)]}), explanation indicates {correct_options_from_exp} ({[option_map[c] for c in correct_options_from_exp]})")

    correct_answer_text = ", ".join(correct_answers) if correct_answers else "Unknown"

    # Create text chunk with sub-options for multi-choice or "All are true"
    text = [f"Question: {question}"]
    if sub_options and (choice_type == "multi" or opts["D"].lower() == "all are true"):
        text.append(f"Correct Answer{'s' if len(correct_answers) > 1 else ''}: {correct_answer_text}")
    else:
        text.append(f"Correct Answer{'s' if len(correct_answers) > 1 else ''}: {correct_answer_text}")
    text.append(f"Explanation: {explanation}")

    ids.append(qid)
    texts.append("\n".join(text))
    metadata = {
        "text": "\n".join(text),
        "correct_options": [str(c) for c in correct_options],
        "correct_answer_text": correct_answer_text,
        "option_a": opts["A"],
        "option_b": opts["B"],
        "option_c": opts["C"],
        "option_d": opts["D"],
        "subject_name": item["subject_name"] if item["subject_name"] is not None else "Unknown",
        "topic_name": item["topic_name"] if item["topic_name"] is not None else "Unknown",
        "choice_type": choice_type
    }
    # Add sub-options to metadata
    for i, (num, text) in enumerate(sub_options.items()):
        metadata[f"sub_option_{i+1}"] = text
    metadatas.append(metadata)

logger.info(f"Prepared {len(texts)} text chunks for embedding.")

# Initialize embedder
try:
    embedder = HuggingFaceEmbeddings(
        model_name="bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
except Exception as e:
    logger.error(f"Failed to initialize embedder: {e}")
    raise

batch_size = 32  # For 768-dim model and 8GB VRAM
vectors = []

logger.info("Generating embeddings in batches with mixed precision...")
for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
    batch_texts = texts[i : i + batch_size]
    try:
        with autocast('cuda'):
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
        (ids[j], vectors[j], metadatas[j])
        for j in range(i, min(i + batch_size, len(vectors)))
    ]
    try:
        index.upsert(vectors=to_up, namespace="")
    except Exception as e:
        logger.error(f"Error upserting batch {i // batch_size}: {e}")
        continue
logger.info(f"Upsert complete. Indexed {len(vectors)} vectors.")
