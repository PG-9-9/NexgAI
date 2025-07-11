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
from cohere import Client as CohereClient
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI

from src.prompt import system_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# load API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY   = os.getenv("COHERE_API_KEY")
if not (PINECONE_API_KEY and OPENAI_API_KEY and COHERE_API_KEY):
    raise RuntimeError("Set PINECONE_API_KEY, OPENAI_API_KEY, COHERE_API_KEY in .env")

# init OpenAI client (v1.0+)
client = OpenAI(api_key=OPENAI_API_KEY)

# detect device for embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Embedding device: {device}")

# init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=os.getenv("PINECONE_ENVIRONMENT","us-east1-gcp"))
INDEX_NAME = "medicalbot"
if INDEX_NAME not in pc.list_indexes().names():
    raise RuntimeError(f"Index '{INDEX_NAME}' not found; run store_index.py first")

# summary‐buffer memory for long chats
memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=3000
)

# shared BGE-large embedder
embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device":device},
    encode_kwargs={"normalize_embeddings":True},
)

# Cohere reranker
co = CohereClient(api_key=COHERE_API_KEY)

from typing import TypedDict, List, Dict, Any

class ChatState(TypedDict):
    query: str
    index_name: str
    namespace: str
    chat_history: List[Dict[str,Any]]
    docs: List[Dict[str,Any]]
    response: str

# dense retrieval node
def retrieve_dense(state: ChatState) -> dict:
    vec = embedder.embed_query(state["query"])
    idx = pc.Index(state["index_name"])
    resp = idx.query(vector=vec, top_k=5, include_metadata=True, namespace=state["namespace"])
    return {"docs":[{"id":m["id"],"text":m["metadata"]["text"]} for m in resp["matches"]]}

# HyDE fallback if no dense hits
def hyde_fallback(state: ChatState) -> dict:
    if state["docs"]:
        return {}
    prompt = f"Draft a plausible answer to: {state['query']}"
    hypo = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role":"user","content":prompt}]
    )
    text = hypo.choices[0].message.content
    vec = embedder.embed_query(text)
    idx = pc.Index(state["index_name"])
    resp = idx.query(vector=vec, top_k=5, include_metadata=True, namespace=state["namespace"])
    return {"docs":[{"id":m["id"],"text":m["metadata"]["text"]} for m in resp["matches"]]}

# **Fixed rerank node**  
def rerank(state: ChatState) -> dict:
    docs = state["docs"]
    if not docs:
        return {"docs": []}

    texts = [d["text"] for d in docs]
    resp = co.rerank(query=state["query"], documents=texts, top_n=3, model="rerank-english-v3.0")

    reranked = []
    # Cohere Python SDK v5+: resp.results is a list of RerankResult(index, relevance_score, document)
    if hasattr(resp, "results"):
        for result in resp.results:
            idx = result.index
            reranked.append(docs[idx])
    else:
        # older/coarser API returns list of document strings
        text_to_doc = {d["text"]: d for d in docs}
        for doc_text in resp:
            if doc_text in text_to_doc:
                reranked.append(text_to_doc[doc_text])
                if len(reranked) >= 3:
                    break

    return {"docs": reranked}

# final answer generation
def refine_response(state: ChatState) -> dict:
    messages = [{"role":"system","content":system_prompt}]
    for m in state["chat_history"]:
        role = "user" if m["role"]=="human" else "assistant"
        messages.append({"role":role,"content":m["content"]})
    if state["docs"]:
        snippet = "\n\n".join(d["text"] for d in state["docs"])
        messages.append({"role":"system","content":f"Relevant excerpts:\n{snippet}"})
    messages.append({"role":"user","content":state["query"]})

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.4, max_tokens=500
    )
    return {"response":resp.choices[0].message.content.strip()}

# assemble LangGraph
graph = StateGraph(ChatState)
graph.add_node("retrieve", retrieve_dense)
graph.add_node("hyde", hyde_fallback)
graph.add_node("rerank", rerank)
graph.add_node("refine", refine_response)

graph.add_conditional_edges(
    "retrieve",
    lambda s: "hyde" if not s["docs"] else "rerank",
    {"hyde":"hyde","rerank":"rerank"}
)
graph.add_edge("hyde","rerank")
graph.add_edge("rerank","refine")
graph.set_entry_point("retrieve")
graph.set_finish_point("refine")

compiled = graph.compile()

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    memory.clear()
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    user = request.form["msg"]
    logger.info("User: %s", user)

    hist = memory.load_memory_variables({})["chat_history"]
    history = [{"role":m.type=="human" and "human" or "assistant","content":m.content} for m in hist]

    state:ChatState = {
        "query": user,
        "index_name": INDEX_NAME,
        "namespace": "",
        "chat_history": history,
        "docs": [],
        "response": ""
    }
    result = compiled.invoke(state)
    answer = result["response"]
    logger.info("Bot: %s", answer)

    memory.save_context({"input":user},{"output":answer})
    return answer

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
