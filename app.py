# app.py

from dotenv import load_dotenv
load_dotenv()

import os
import time
import logging
import torch

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware

from pinecone import Pinecone
from langgraph.graph import StateGraph
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from cohere import Client as CohereClient
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai.chat_models import ChatOpenAI

from tenacity import retry, wait_exponential, stop_after_attempt
from cachetools import TTLCache

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.prompt import system_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Prometheus metrics
REQ_COUNT    = Counter("requests_total", "Total chat requests")
LATENCY_H    = Histogram("request_latency_seconds", "Latency of /get")
BRANCH_COUNT = Counter("branch", "Branches taken", ["step"])

# Config & keys
INDEX_NAME = os.getenv("PINECONE_INDEX", "medicalbot")
SIM_THRESH  = float(os.getenv("SIM_THRESH", "0.65"))
CACHE_TTL   = int(os.getenv("CACHE_TTL", "600"))
RATE_LIMIT  = int(os.getenv("RATE_LIMIT", "30"))

pc_key   = os.getenv("PINECONE_API_KEY")
open_key = os.getenv("OPENAI_API_KEY")
co_key   = os.getenv("COHERE_API_KEY", "")

if not (pc_key and open_key):
    raise RuntimeError("PINECONE_API_KEY & OPENAI_API_KEY required")

co     = CohereClient(api_key=co_key) if co_key else None
openai = OpenAI(api_key=open_key)
pc     = Pinecone(api_key=pc_key, environment=os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp"))

if INDEX_NAME not in pc.list_indexes().names():
    raise RuntimeError(f"Index '{INDEX_NAME}' not found; run store_index.py first")

# Memory & embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Embedding on device: {device}")

memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=3000
)

embedder = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

# Cache & rate-limiter
cache = TTLCache(maxsize=1000, ttl=CACHE_TTL)
rate_windows = {}

# Retry wrappers
@retry(wait=wait_exponential(0.5, 4), stop=stop_after_attempt(3))
def pinecone_query(vec, top_k):
    return pc.Index(INDEX_NAME).query(vector=vec, top_k=top_k, include_metadata=True, namespace="")

@retry(wait=wait_exponential(0.5, 4), stop=stop_after_attempt(3))
def cohere_rerank(query, docs):
    return co.rerank(query=query, documents=docs, top_n=3, model="rerank-english-v3.0")

# LangGraph state
from typing import TypedDict, List, Dict, Any
class ChatState(TypedDict):
    query: str
    index_name: str
    namespace: str
    chat_history: List[Dict[str, Any]]
    docs: List[Dict[str, Any]]
    response: str

# Graph nodes
def retrieve(state: ChatState) -> dict:
    BRANCH_COUNT.labels(step="retrieve").inc()
    vec = embedder.embed_query(state["query"])
    resp = pinecone_query(vec, 5)
    docs = [{"id": m["id"], "text": m["metadata"]["text"], "score": m["score"]} for m in resp["matches"]]
    return {"docs": docs}

def hyde(state: ChatState) -> dict:
    BRANCH_COUNT.labels(step="hyde").inc()
    docs = state["docs"]
    if docs and docs[0]["score"] >= SIM_THRESH:
        return {}
    prompt = f"Draft a plausible answer to: {state['query']}"
    hypo = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    text = hypo.choices[0].message.content
    vec = embedder.embed_query(text)
    resp = pinecone_query(vec, 5)
    return {"docs": [{"id": m["id"], "text": m["metadata"]["text"], "score": m["score"]} for m in resp["matches"]]}

def rerank(state: ChatState) -> dict:
    BRANCH_COUNT.labels(step="rerank").inc()
    docs = state["docs"]
    if not docs or co is None:
        return {"docs": docs}
    texts = [d["text"] for d in docs]
    rr = cohere_rerank(state["query"], texts)
    ranked = []
    if hasattr(rr, "results"):
        for r in rr.results:
            ranked.append(docs[r.index])
    else:
        mp = {d["text"]: d for d in docs}
        for t in rr:
            if t in mp:
                ranked.append(mp[t])
                if len(ranked) >= 3:
                    break
    return {"docs": ranked}

def refine(state: ChatState) -> dict:
    BRANCH_COUNT.labels(step="refine").inc()
    messages = [{"role":"system","content":system_prompt}]
    for m in state["chat_history"]:
        role = "user" if m["role"]=="human" else "assistant"
        messages.append({"role": role, "content": m["content"]})
    if state["docs"]:
        snippet = "\n\n".join(d["text"] for d in state["docs"])
        messages.append({"role":"system","content":f"Relevant excerpts:\n{snippet}"})
    messages.append({"role":"user","content":state["query"]})

    # synchronous ChatCompletion
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.4,
        max_tokens=500
    )
    answer = resp.choices[0].message.content.strip()
    return {"response": answer}

# Assemble graph
graph = StateGraph(ChatState)
graph.add_node("retrieve", retrieve)
graph.add_node("hyde", hyde)
graph.add_node("rerank", rerank)
graph.add_node("refine", refine)
graph.add_conditional_edges(
    "retrieve",
    lambda s: "hyde" if not s["docs"] or s["docs"][0]["score"] < SIM_THRESH else "rerank",
    {"hyde": "hyde", "rerank": "rerank"}
)
graph.add_edge("hyde", "rerank")
graph.add_edge("rerank", "refine")
graph.set_entry_point("retrieve")
graph.set_finish_point("refine")
compiled = graph.compile()

# FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/metrics")
def metrics():
    return HTMLResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    logger.info("Rendering chat UI")
    memory.clear()
    return templates.TemplateResponse("chat.html", {"request": request})

def check_rate(ip: str) -> bool:
    now = int(time.time())
    window = rate_windows.setdefault(ip, [])
    window = [t for t in window if t > now - 60]
    if len(window) >= RATE_LIMIT:
        return False
    window.append(now)
    rate_windows[ip] = window
    return True

@app.post("/get")
async def chat(request: Request):
    REQ_COUNT.inc()
    client_ip = request.client.host
    if not check_rate(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests")

    form = await request.form()
    msg = form.get("msg", "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Empty message")

    if msg in cache:
        docs, history = cache[msg]
    else:
        hist = memory.load_memory_variables({})["chat_history"]
        history = [{"role": "user" if m.type == "human" else "assistant", "content": m.content} for m in hist]
        state = {
            "query": msg,
            "index_name": INDEX_NAME,
            "namespace": "",
            "chat_history": history,
            "docs": [],
            "response": ""
        }
        out = compiled.invoke(state)
        answer = out["response"]
        docs = state["docs"]
        cache[msg] = (docs, history)
        memory.save_context({"input": msg}, {"output": answer})

    # Return plain text response
    return JSONResponse(content=answer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
