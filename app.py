# app.py

from dotenv import load_dotenv
load_dotenv()

import os
import logging
import torch

from flask import Flask, render_template, request
from pinecone import Pinecone
from pinecone.exceptions import PineconeException
from langgraph.graph import StateGraph
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

from src.prompt import system_prompt
from langchain.memory import ConversationBufferMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# load and validate keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
if not (PINECONE_API_KEY and OPENAI_API_KEY):
    raise RuntimeError("Missing PINECONE_API_KEY or OPENAI_API_KEY in .env")

# initialize OpenAI client for v1.0.0+
client = OpenAI(api_key=OPENAI_API_KEY)

# detect GPU vs CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device for embeddings: {device}")

# initialize Pinecone
pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
)
INDEX_NAME = "medicalbot"
if INDEX_NAME not in pc.list_indexes().names():
    raise RuntimeError(f"Pinecone index '{INDEX_NAME}' not found – run store_index.py first")

# in-memory conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# embedder matching your stored vectors
embedder = HuggingFaceEmbeddings(
    model_name="bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

# state schema
from typing import TypedDict, List, Dict, Any

class ChatState(TypedDict):
    query: str
    index_name: str
    namespace: str
    chat_history: List[Dict[str, Any]]
    docs: List[Dict[str, Any]]
    response: str

# LangGraph node: retrieve from Pinecone
def pinecone_tool(state: ChatState) -> dict:
    q_vec = embedder.embed_query(state["query"])
    idx = pc.Index(state["index_name"])
    resp = idx.query(
        vector=q_vec,
        top_k=3,
        include_metadata=True,
        namespace=state["namespace"]
    )
    docs = []
    for match in resp["matches"]:
        docs.append({"id": match["id"], "text": match["metadata"].get("text", "")})
    return {"docs": docs}

# LangGraph node: call the new OpenAI client
def refine_response(state: ChatState) -> dict:
    messages = [{"role": "system", "content": system_prompt}]
    for m in state["chat_history"]:
        role = "user" if m["role"] == "human" else "assistant"
        messages.append({"role": role, "content": m["content"]})
    if state["docs"]:
        snippet = "\n\n".join(d["text"] for d in state["docs"])
        messages.append({
            "role": "system",
            "content": f"Relevant excerpts:\n{snippet}"
        })
    messages.append({"role": "user", "content": state["query"]})

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.4,
        max_tokens=500
    )
    # choices is a list of ChatCompletionChoice objects
    answer = resp.choices[0].message.content.strip()
    return {"response": answer}

# assemble the graph
graph = StateGraph(ChatState)
graph.add_node("pinecone", pinecone_tool)
graph.add_node("refine", refine_response)
graph.add_edge("pinecone", "refine")
graph.set_entry_point("pinecone")
graph.set_finish_point("refine")
compiled = graph.compile()

# Flask app
app = Flask(__name__)

@app.route("/")
def index():
    memory.clear()
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    user_msg = request.form["msg"]
    logger.info("User: %s", user_msg)

    hist = memory.load_memory_variables({})["chat_history"]
    history_formatted = [
        {"role": m.type == "human" and "human" or "assistant", "content": m.content}
        for m in hist
    ]

    state: ChatState = {
        "query": user_msg,
        "index_name": INDEX_NAME,
        "namespace": "",
        "chat_history": history_formatted,
        "docs": [],
        "response": ""
    }
    final = compiled.invoke(state)
    answer = final["response"]
    logger.info("Bot: %s", answer)

    memory.save_context({"input": user_msg}, {"output": answer})
    return answer

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
