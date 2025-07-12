from dotenv import load_dotenv
load_dotenv()

import os, time, logging, torch, json, re
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

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

# Metrics
REQ_COUNT    = Counter("requests_total", "Total chat requests")
LATENCY_H    = Histogram("request_latency_seconds", "Latency of /get")
BRANCH_COUNT = Counter("branch", "Branches taken", ["step"])

# Config
INDEX_NAME = os.getenv("PINECONE_INDEX", "medicalbot")
SIM_THRESH = float(os.getenv("SIM_THRESH", "0.65"))
CACHE_TTL  = int(os.getenv("CACHE_TTL", "600"))
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "30"))

pc_key   = os.getenv("PINECONE_API_KEY")
open_key = os.getenv("OPENAI_API_KEY")
co_key   = os.getenv("COHERE_API_KEY", "")

if not (pc_key and open_key):
    raise RuntimeError("PINECONE_API_KEY & OPENAI_API_KEY required")

co     = CohereClient(api_key=co_key) if co_key else None
openai = OpenAI(api_key=open_key)
pc     = Pinecone(api_key=pc_key, environment=os.getenv("PINECONE_ENVIRONMENT","us-east1-gcp"))

if INDEX_NAME not in pc.list_indexes().names():
    raise RuntimeError(f"Index '{INDEX_NAME}' not found; run store_index.py first")

# Memory & Embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Embedding on device: {device}")

memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    memory_key="chat_history", return_messages=True, max_token_limit=3000
)

embedder = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL","BAAI/bge-large-en-v1.5"),
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings":True},
)

# Cache & rate limiter
cache = TTLCache(maxsize=1000, ttl=CACHE_TTL)
rate_windows = {}
def check_rate(ip:str)->bool:
    now = int(time.time())
    win = rate_windows.setdefault(ip,[])
    win = [t for t in win if t>now-60]
    if len(win)>=RATE_LIMIT: return False
    win.append(now); rate_windows[ip]=win
    return True

# Retry wrappers
@retry(wait=wait_exponential(0.5,4), stop=stop_after_attempt(3))
def pinecone_query(vec, top_k):
    return pc.Index(INDEX_NAME).query(vector=vec, top_k=top_k, include_metadata=True, namespace="")

@retry(wait=wait_exponential(0.5,4), stop=stop_after_attempt(3))
def cohere_rerank(query, docs):
    return co.rerank(query=query, documents=docs, top_n=3, model="rerank-english-v3.0")

# Dashboard spec for Grafana
grafana_dashboard = {
  "title":"MedBot Metrics",
  "panels":[
    {"type":"graph","title":"Requests/sec","datasource":"Prometheus","targets":[{"expr":"rate(requests_total[1m])"}]},
    {"type":"graph","title":"Latency p50","datasource":"Prometheus","targets":[{"expr":"histogram_quantile(0.5, sum(rate(request_latency_seconds_bucket[5m])) by (le))"}]},
    {"type":"graph","title":"Branch counts","datasource":"Prometheus","targets":[{"expr":"sum(branch) by (step)"}]}
  ]
}

# LangGraph pipeline
from typing import TypedDict, List, Dict, Any
class ChatState(TypedDict):
    query: str; index_name: str; namespace: str
    chat_history: List[Dict[str,Any]]; docs: List[Dict[str,Any]]; response: str

def retrieve(state: ChatState)->dict:
    BRANCH_COUNT.labels(step="retrieve").inc()
    vec  = embedder.embed_query(state["query"])
    resp = pinecone_query(vec,5)
    docs = [{"id":m["id"],"text":m["metadata"]["text"],"score":m["score"]} for m in resp["matches"]]
    return {"docs":docs}

def hyde(state: ChatState)->dict:
    BRANCH_COUNT.labels(step="hyde").inc()
    docs=state["docs"]
    if docs and docs[0]["score"]>=SIM_THRESH: return {}
    prompt=f"Draft a plausible answer to: {state['query']}"
    hypo=openai.chat.completions.create(model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],temperature=0.0
    )
    text=hypo.choices[0].message.content
    vec=embedder.embed_query(text)
    resp=pinecone_query(vec,5)
    return {"docs":[{"id":m["id"],"text":m["metadata"]["text"],"score":m["score"]} for m in resp["matches"]]}

def rerank(state: ChatState)->dict:
    BRANCH_COUNT.labels(step="rerank").inc()
    docs=state["docs"]
    if not docs or co is None:
        logger.info("Rerank skipped")
        return {"docs":docs}
    texts=[d["text"] for d in docs]
    rr=cohere_rerank(state["query"],texts)
    ranked=[]
    if hasattr(rr,"results"):
        for r in rr.results: ranked.append(docs[r.index])
    else:
        mp={d["text"]:d for d in docs}
        for t in rr:
            if t in mp:
                ranked.append(mp[t])
                if len(ranked)>=3: break
    logger.info(f"Rerank returned {len(ranked)} docs")
    return {"docs":ranked}

def refine(state: ChatState)->dict:
    BRANCH_COUNT.labels(step="refine").inc()
    msgs=[{"role":"system","content":system_prompt}]
    for m in state["chat_history"]:
        role="user" if m["role"]=="human" else "assistant"
        msgs.append({"role":role,"content":m["content"]})
    if state["docs"]:
        snippet="\n\n".join(d["text"] for d in state["docs"])
        msgs.append({"role":"system","content":f"Relevant excerpts:\n{snippet}"})
    msgs.append({"role":"user","content":state["query"]})

    resp=openai.chat.completions.create(
        model="gpt-3.5-turbo",messages=msgs,temperature=0.4,max_tokens=500
    )
    return {"response":resp.choices[0].message.content.strip()}

graph=StateGraph(ChatState)
graph.add_node("retrieve",retrieve)
graph.add_node("hyde",hyde)
graph.add_node("rerank",rerank)
graph.add_node("refine",refine)
graph.add_conditional_edges("retrieve",
    lambda s:"hyde" if not s["docs"] or s["docs"][0]["score"]<SIM_THRESH else "rerank",
    {"hyde":"hyde","rerank":"rerank"}
)
graph.add_edge("hyde","rerank")
graph.add_edge("rerank","refine")
graph.set_entry_point("retrieve")
graph.set_finish_point("refine")
compiled=graph.compile()

# Datetime formatting for Jinja2 filter
def datetime_format(value, format='%B %d, %Y, %I:%M %p %Z'):
    if value is None or not isinstance(value, (int, float)):
        return 'N/A'
    return datetime.fromtimestamp(value).strftime(format)

# FastAPI setup
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Registering the custom filter
templates.env.filters['datetime_format'] = datetime_format

@app.get("/metrics")
def metrics():                  # Prometheus scraper
    return HTMLResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():                   # raw JSON
    return JSONResponse({"status":"ok","time":time.time()})

#TODO: Add a route to serve the Grafana dashboard JSON, (Not implemented for now)
@app.get("/dashboard.json")
def dashboard_json():           # raw JSON
    return JSONResponse(grafana_dashboard)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    memory.clear()
    return templates.TemplateResponse("chat.html", {"request":request})

# -- Other UI routes --

@app.get("/health-ui", response_class=HTMLResponse)
def health_ui(request: Request):
    data = {"status":"ok","time":time.time()}
    return templates.TemplateResponse("health.html", {"request":request, **data})

@app.get("/metrics-ui", response_class=HTMLResponse)
def metrics_ui(request: Request):
    txt = generate_latest().decode("utf-8")
    logger.info(f"Metrics Text: {txt}")  # Debug metrics
    
    # Parse Prometheus metrics
    metrics = {}
    lines = txt.split('\n')
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        match = re.match(r'^(\w+)(?:{([^}]*)})?\s+([\d\.e\+-]+)$', line)
        if match:
            name, labels, value = match.groups()
            if not metrics.get(name):
                metrics[name] = []
            metrics[name].append({"labels": labels or "", "value": float(value)})
    
    # Extract specific metrics for the template
    context = {
        "request": request,
        "metrics_text": txt,
        "requests_total": next((m["value"] for m in metrics.get("requests_total", [])), "N/A"),
        "requests_created": next((m["value"] for m in metrics.get("requests_created", [])), "N/A"),
        "request_latency_seconds_created": next((m["value"] for m in metrics.get("request_latency_seconds_created", [])), "N/A"),
        "branch_created_retrieve": next((m["value"] for m in metrics.get("branch_created", []) if 'step="retrieve"' in m["labels"]), "N/A"),
        "branch_created_rerank": next((m["value"] for m in metrics.get("branch_created", []) if 'step="rerank"' in m["labels"]), "N/A"),
        "branch_created_refine": next((m["value"] for m in metrics.get("branch_created", []) if 'step="refine"' in m["labels"]), "N/A"),
        "gc_objects_collected": [
            next((m["value"] for m in metrics.get("python_gc_objects_collected_total", []) if 'generation="0"' in m["labels"]), 0),
            next((m["value"] for m in metrics.get("python_gc_objects_collected_total", []) if 'generation="1"' in m["labels"]), 0),
            next((m["value"] for m in metrics.get("python_gc_objects_collected_total", []) if 'generation="2"' in m["labels"]), 0),
        ],
        "gc_collections": [
            next((m["value"] for m in metrics.get("python_gc_collections_total", []) if 'generation="0"' in m["labels"]), 0),
            next((m["value"] for m in metrics.get("python_gc_collections_total", []) if 'generation="1"' in m["labels"]), 0),
            next((m["value"] for m in metrics.get("python_gc_collections_total", []) if 'generation="2"' in m["labels"]), 0),
        ],
        "uncollectable_objects": [
            next((m["value"] for m in metrics.get("python_gc_objects_uncollectable_total", []) if 'generation="0"' in m["labels"]), 0),
            next((m["value"] for m in metrics.get("python_gc_objects_uncollectable_total", []) if 'generation="1"' in m["labels"]), 0),
            next((m["value"] for m in metrics.get("python_gc_objects_uncollectable_total", []) if 'generation="2"' in m["labels"]), 0),
        ],
        "latency_buckets": [
            next((m["value"] for m in metrics.get("request_latency_seconds_bucket", []) if f'le="{le}"' in m["labels"]), 0)
            for le in ["0.005", "0.01", "0.025", "0.05", "0.075", "0.1", "0.25", "0.5", "0.75", "1.0", "2.5", "5.0", "7.5", "10.0", "+Inf"]
        ],
        "branch_total": [
            next((m["value"] for m in metrics.get("branch_total", []) if 'step="retrieve"' in m["labels"]), 0),
            next((m["value"] for m in metrics.get("branch_total", []) if 'step="rerank"' in m["labels"]), 0),
            next((m["value"] for m in metrics.get("branch_total", []) if 'step="refine"' in m["labels"]), 0),
        ]
    }
    
    response = templates.TemplateResponse("metrics.html", context)
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; connect-src 'self'"
    return response

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_ui(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request":request, "dashboard": grafana_dashboard})

@app.post("/get")
async def chat(request: Request):
    REQ_COUNT.inc()
    ip=request.client.host
    if not check_rate(ip):
        raise HTTPException(429,"Too many requests")

    form=await request.form()
    msg=form.get("msg","").strip()
    if not msg:
        raise HTTPException(400,"Empty message")

    hist=memory.load_memory_variables({})["chat_history"]
    history=[{"role":"user" if m.type=="human" else "assistant","content":m.content} for m in hist]
    state={"query":msg,"index_name":INDEX_NAME,"namespace":"","chat_history":history,"docs":[],"response":""}
    out=await run_in_threadpool(compiled.invoke, state)
    answer=out["response"]
    memory.save_context({"input":msg},{"output":answer})
    return JSONResponse(content=answer)

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data=await ws.receive_json()
            msg=data.get("msg","").strip()
            if not msg:
                await ws.send_json({"error":"Empty message"}); continue
            hist=memory.load_memory_variables({})["chat_history"]
            history=[{"role":"user" if m.type=="human" else "assistant","content":m.content} for m in hist]
            state={"query":msg,"index_name":INDEX_NAME,"namespace":"","chat_history":history,"docs":[],"response":""}
            out=await run_in_threadpool(compiled.invoke, state)
            ans=out["response"]
            memory.save_context({"input":msg},{"output":ans})
            await ws.send_json({"response":ans})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

if __name__=="__main__":
    import uvicorn
    uvicorn.run("app:app",host="0.0.0.0",port=8080,reload=True)