from dotenv import load_dotenv
load_dotenv()

import os
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

from src.prompt import system_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Load and validate keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY   = os.getenv("COHERE_API_KEY", "")

if not (PINECONE_API_KEY and OPENAI_API_KEY):
    raise RuntimeError("Set PINECONE_API_KEY & OPENAI_API_KEY in .env")

if not COHERE_API_KEY:
    logger.warning("COHERE_API_KEY not set; rerank disabled")
    co = None
else:
    co = CohereClient(api_key=COHERE_API_KEY)

# Init clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc     = Pinecone(api_key=PINECONE_API_KEY,
                  environment=os.getenv("PINECONE_ENVIRONMENT","us-east1-gcp"))
INDEX_NAME = "medicalbot"
if INDEX_NAME not in pc.list_indexes().names():
    raise RuntimeError(f"Index '{INDEX_NAME}' not found – run store_index.py first")

# Memory & embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=3000
)
embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

# LangGraph state definition
from typing import TypedDict, List, Dict, Any
class ChatState(TypedDict):
    query: str
    index_name: str
    namespace: str
    chat_history: List[Dict[str,Any]]
    docs: List[Dict[str,Any]]
    response: str

# LangGraph nodes
def retrieve(state: ChatState) -> dict:
    vec = embedder.embed_query(state["query"])
    idx = pc.Index(state["index_name"])
    resp = idx.query(vector=vec, top_k=5, include_metadata=True, namespace=state["namespace"])
    return {"docs":[{"id":m["id"],"text":m["metadata"]["text"],"score":m["score"]} for m in resp["matches"]]}

def hyde(state: ChatState) -> dict:
    docs = state["docs"]
    if docs and docs[0]["score"] >= 0.65:
        return {}
    prompt = f"Draft a plausible answer to: {state['query']}"
    hypo = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0
    )
    text = hypo.choices[0].message.content
    vec = embedder.embed_query(text)
    idx = pc.Index(state["index_name"])
    resp = idx.query(vector=vec, top_k=5, include_metadata=True, namespace=state["namespace"])
    return {"docs":[{"id":m["id"],"text":m["metadata"]["text"],"score":m["score"]} for m in resp["matches"]]}

def rerank(state: ChatState) -> dict:
    docs = state["docs"]
    if not docs or co is None:
        return {"docs": docs}
    texts = [d["text"] for d in docs]
    rr = co.rerank(query=state["query"], documents=texts, top_n=3, model="rerank-english-v3.0")
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
    return {"response": resp.choices[0].message.content.strip()}

# Assemble LangGraph
graph = StateGraph(ChatState)
graph.add_node("retrieve", retrieve)
graph.add_node("hyde", hyde)
graph.add_node("rerank", rerank)
graph.add_node("refine", refine)
graph.add_conditional_edges(
    "retrieve",
    lambda s: "hyde" if not s["docs"] or s["docs"][0]["score"] < 0.65 else "rerank",
    {"hyde":"hyde","rerank":"rerank"}
)
graph.add_edge("hyde","rerank")
graph.add_edge("rerank","refine")
graph.set_entry_point("retrieve")
graph.set_finish_point("refine")
compiled = graph.compile()

# FastAPI app + static files + templates
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    memory.clear()
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/get")
async def chat(request: Request):
    form = await request.form()
    user_msg = form.get("msg","").strip()
    if not user_msg:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty message")

    hist = memory.load_memory_variables({})["chat_history"]
    history = [{"role": m.type=="human" and "human" or "assistant", "content": m.content} for m in hist]

    state: ChatState = {
        "query": user_msg,
        "index_name": INDEX_NAME,
        "namespace": "",
        "chat_history": history,
        "docs": [],
        "response": ""
    }
    out = compiled.invoke(state)
    ans = out["response"]
    memory.save_context({"input":user_msg}, {"output":ans})
    return JSONResponse(content=ans)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
