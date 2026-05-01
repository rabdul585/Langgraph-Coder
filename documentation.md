# LangGraph Code Review Pipeline
## Complete Technical Documentation

**Version:** 1.0.0 | **Last Updated:** May 2025 | **Stack:** LangGraph · OpenRouter · Streamlit · Python 3.11+

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Component Deep Dive](#3-component-deep-dive)
4. [State Machine & Conditional Edges](#4-state-machine--conditional-edges)
5. [File Structure](#5-file-structure)
6. [Local Setup & Running](#6-local-setup--running)
7. [Scaling to Production](#7-scaling-to-production)
8. [Prompt Version Management](#8-prompt-version-management)
9. [Latency Optimization](#9-latency-optimization)
10. [Observability & Monitoring](#10-observability--monitoring)
11. [Security Best Practices](#11-security-best-practices)
12. [Testing Strategy](#12-testing-strategy)
13. [Cost Management](#13-cost-management)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. System Overview

The LangGraph Code Review Pipeline is a **self-healing, multi-agent AI system** that:

1. Takes a plain-English feature request
2. Generates production-quality Python code (Node 1)
3. Writes a comprehensive pytest test suite (Node 2)
4. Executes tests in an isolated subprocess (Node 3)
5. Routes through **conditional edges** — if tests fail, a Code Fixer node learns from the errors and rewrites the code, then loops back to testing
6. Reviews code quality with a senior reviewer agent (Node 4)
7. Routes again — if rejected, a Code Improver node addresses the review feedback and loops back
8. Exits with a final approved artifact or a detailed failure report

The key innovation is the **feedback loop**: the system does not stop on failure. It reads the exact error output, understands what went wrong, and autonomously corrects its own work — up to `MAX_RETRIES = 3` times per failure type.

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Separation of concerns** | `app.py` = pure pipeline logic. `streamlit_app.py` = UI only. |
| **Shared state, not passed parameters** | `CodeReviewState` TypedDict flows through all nodes |
| **Fail-safe loops** | `MAX_RETRIES` cap prevents infinite loops |
| **Secure credential handling** | API key read fresh per call, never stored globally |
| **Import-safe** | `graph.invoke()` only inside `if __name__ == "__main__"` |

---

## 2. Architecture Diagram

### 2.1 Full Pipeline Flow (with Retry Loops)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER INPUT                                        │
│              "Create a function to validate emails"                      │
│                     model_key = "GPT-OSS 120B"                          │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  START         │
                    └───────┬────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │  🤖 NODE 1                  │
              │  code_generator_node        │
              │                             │
              │  reads  ← feature_request   │
              │            model_key        │
              │  writes → code_content      │
              │                             │
              │  [LLM: OpenRouter]          │
              └─────────────┬───────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │  🧪 NODE 2                                    │◄──────────────────┐
     │  test_generator_node                          │                   │
     │                                              │                   │
     │  reads  ← code_content                       │                   │
     │            feature_request                   │                   │
     │  writes → test_content                       │                   │
     │                                              │                   │
     │  [LLM: OpenRouter]                           │                   │
     └──────────────────────┬───────────────────────┘                   │
                            │                                            │
                            ▼                                            │
     ┌──────────────────────────────────────────────┐                   │
     │  ▶️  NODE 3                                    │                   │
     │  test_runner_node                             │                   │
     │                                              │                   │
     │  reads  ← code_content                       │                   │
     │            test_content                      │                   │
     │  writes → test_results                       │                   │
     │            tests_passed                      │                   │
     │                                              │                   │
     │  [subprocess: pytest]  — NO LLM             │                   │
     └──────────────────────┬───────────────────────┘                   │
                            │                                            │
                            ▼                                            │
              ┌─────────────────────────┐                               │
              │  route_after_tests()    │  ← CONDITIONAL EDGE           │
              │  (decision function)    │                               │
              └──────┬──────────┬───────┘                               │
                     │          │                                        │
               PASS  │          │ FAIL (retries left)                   │
                     │          │                                        │
                     │          ▼                                        │
                     │  ┌──────────────────────────────────────────┐    │
                     │  │  🔧 NODE 5                               │    │
                     │  │  code_fixer_node                         │    │
                     │  │                                          │    │
                     │  │  reads  ← code_content                   │    │
                     │  │            test_results (failures)       │    │
                     │  │  writes → code_content (fixed)           │    │
                     │  │            retry_count + 1               │    │
                     │  │            fix_history (appended)        │    │
                     │  │                                          │    │
                     │  │  [LLM: learns from error output]        │    │
                     │  └──────────────────┬───────────────────────┘    │
                     │                     │                             │
                     │                     └─────────────────────────────┘
                     │                     (loops back to Node 2)
                     │
                     ▼
     ┌──────────────────────────────────────────────┐
     │  👀 NODE 4                                    │
     │  code_reviewer_node                           │
     │                                              │
     │  reads  ← code_content                       │
     │            test_results                      │
     │            tests_passed                      │
     │  writes → review_feedback                    │
     │            is_approved                       │
     │                                              │
     │  [LLM: OpenRouter]                           │
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  route_after_review()   │  ← CONDITIONAL EDGE
              │  (decision function)    │
              └──────┬──────────┬───────┘
                     │          │
              APPROVED│          │ REJECTED (retries left)
                     │          │
                     │          ▼
                     │  ┌──────────────────────────────────────────┐
                     │  │  ✨ NODE 6                               │
                     │  │  code_improver_node                      │
                     │  │                                          │
                     │  │  reads  ← code_content                   │
                     │  │            review_feedback               │
                     │  │  writes → code_content (improved)        │
                     │  │            retry_count + 1               │
                     │  │            fix_history (appended)        │
                     │  │                                          │
                     │  │  [LLM: learns from review feedback]     │
                     │  └──────────────────┬───────────────────────┘
                     │                     │
                     │                     └──► Node 2 (loops back)
                     │
                     ▼
                  ┌──────┐
                  │  END │
                  └──────┘
              Final CodeReviewState
```

### 2.2 System Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                              │
│                    streamlit_app.py                                  │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│   │  Input   │  │  Live    │  │  Result  │  │  Fix History     │  │
│   │  Form    │  │  Node    │  │  Tabs    │  │  Tab             │  │
│   │          │  │  Cards   │  │          │  │                  │  │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │  imports from
┌─────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                             │
│                         app.py                                       │
│                                                                      │
│   CodeReviewState (TypedDict)  ←  shared state flowing through all  │
│                                                                      │
│   6 Node Functions             ←  pure Python, read/write state     │
│   2 Conditional Edge Functions ←  routing logic based on state      │
│   1 Compiled StateGraph        ←  LangGraph directed graph          │
└─────────────────────────────────────────────────────────────────────┘
                              │  calls
┌─────────────────────────────────────────────────────────────────────┐
│                      EXECUTION LAYER                                 │
│                                                                      │
│   ┌──────────────────────────┐   ┌──────────────────────────────┐  │
│   │   OpenRouter API         │   │   Python subprocess          │  │
│   │   (via openai SDK)       │   │   (pytest runner)            │  │
│   │                          │   │                              │  │
│   │   GPT-OSS 120B           │   │   Isolated temp directory    │  │
│   │   Gemini 2.0 Flash       │   │   source.py + test_source.py │  │
│   │   Llama 4 Maverick       │   │   60s timeout                │  │
│   │   DeepSeek R1            │   │                              │  │
│   └──────────────────────────┘   └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │  reads from
┌─────────────────────────────────────────────────────────────────────┐
│                      CONFIGURATION LAYER                             │
│                         .env file                                    │
│                                                                      │
│   OPENROUTER_API_KEY  ←  never hard-coded, never stored globally    │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Production Architecture (Scaled)

```
                        ┌─────────────────────────┐
                        │   Load Balancer          │
                        │   (nginx / AWS ALB)      │
                        └────────────┬────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                       │
              ▼                      ▼                       ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  Streamlit Pod 1 │  │  Streamlit Pod 2 │  │  Streamlit Pod 3 │
    │  (read-only UI)  │  │  (read-only UI)  │  │  (read-only UI)  │
    └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
             │                     │                      │
             └─────────────────────┼──────────────────────┘
                                   │  HTTP / gRPC
                                   ▼
                        ┌─────────────────────────┐
                        │   FastAPI / Pipeline     │
                        │   Service                │
                        │                          │
                        │   POST /run-pipeline     │
                        │   GET  /status/{run_id}  │
                        │   GET  /result/{run_id}  │
                        └────────────┬────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                       │
              ▼                      ▼                       ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  Pipeline        │  │  Pipeline        │  │  Pipeline        │
    │  Worker 1        │  │  Worker 2        │  │  Worker 3        │
    │  (Celery/RQ)     │  │  (Celery/RQ)     │  │  (Celery/RQ)     │
    └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
             │                     │                      │
             ├─────────────────────┼──────────────────────┤
             │                     │                      │
             ▼                     ▼                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              SHARED INFRASTRUCTURE                           │
    │                                                              │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
    │  │  Redis       │  │  PostgreSQL  │  │  Prometheus +    │  │
    │  │  (job queue, │  │  (run state, │  │  Grafana         │  │
    │  │   cache)     │  │   fix hist.) │  │  (metrics)       │  │
    │  └──────────────┘  └──────────────┘  └──────────────────┘  │
    │                                                              │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
    │  │  OpenRouter  │  │  LangSmith   │  │  S3 / Blob       │  │
    │  │  API Gateway │  │  (LLM traces)│  │  (code artifacts)│  │
    │  └──────────────┘  └──────────────┘  └──────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
```

---

## 3. Component Deep Dive

### 3.1 CodeReviewState (The Shared Notepad)

Every node in the pipeline reads from and writes to this single TypedDict. LangGraph merges partial updates back automatically — nodes never need to re-pass fields they did not touch.

```python
class CodeReviewState(TypedDict):
    # ── INPUTS (set once, never modified) ─────────────────────
    feature_request: str   # "Create a function to validate emails"
    model_key:       str   # "GPT-OSS 120B"

    # ── NODE OUTPUTS (filled progressively) ───────────────────
    code_content:    str   # Node 1/5/6 — Python source code
    test_content:    str   # Node 2     — pytest test suite
    test_results:    str   # Node 3     — pytest stdout/stderr
    tests_passed:    bool  # Node 3     — True if pytest exit==0
    review_feedback: str   # Node 4     — LLM prose review
    is_approved:     bool  # Node 4     — True = approved

    # ── RETRY TRACKING ────────────────────────────────────────
    retry_count:     int   # 0→3 (capped at MAX_RETRIES)
    fix_history:     list  # [{attempt, reason, old_code, new_code}]
```

**Why TypedDict instead of a dataclass or Pydantic model?**
LangGraph requires TypedDict. The framework uses it to validate that nodes only update keys that exist in the state schema, and to merge partial update dicts efficiently.

### 3.2 The Six Nodes

| Node | Function | LLM? | Reads | Writes |
|------|----------|-------|-------|--------|
| 1 | `code_generator_node` | ✅ Yes | `feature_request`, `model_key` | `code_content` |
| 2 | `test_generator_node` | ✅ Yes | `code_content`, `feature_request` | `test_content` |
| 3 | `test_runner_node` | ❌ No (subprocess) | `code_content`, `test_content` | `test_results`, `tests_passed` |
| 4 | `code_reviewer_node` | ✅ Yes | `code_content`, `test_results`, `tests_passed` | `review_feedback`, `is_approved` |
| 5 | `code_fixer_node` | ✅ Yes | `code_content`, `test_results` | `code_content`, `retry_count`, `fix_history` |
| 6 | `code_improver_node` | ✅ Yes | `code_content`, `review_feedback` | `code_content`, `retry_count`, `fix_history` |

**Node 3 is the only node with no LLM call.** It runs real `pytest` in a subprocess inside a temporary directory, ensuring code execution is fully isolated from the host environment.

### 3.3 Conditional Edge Functions

```python
# After Node 3 — routes based on test results
def route_after_tests(state) -> "code_reviewer" | "code_fixer" | "__end__":
    if state["tests_passed"]:               return "code_reviewer"   # ← happy path
    elif state["retry_count"] < MAX_RETRIES: return "code_fixer"     # ← retry
    else:                                   return "__end__"          # ← give up

# After Node 4 — routes based on approval
def route_after_review(state) -> "code_improver" | "__end__":
    if state["is_approved"]:                return "__end__"          # ← success
    elif state["retry_count"] < MAX_RETRIES: return "code_improver"  # ← retry
    else:                                   return "__end__"          # ← give up
```

These are **pure functions** — they take state, return a string key, no side effects. LangGraph calls them automatically after the registered source node completes.

---

## 4. State Machine & Conditional Edges

### 4.1 All Possible Execution Paths

```
Path A — Perfect first attempt (no retries):
  START → Node1 → Node2 → Node3[PASS] → Node4[APPROVED] → END

Path B — Test failure, fixed on first retry:
  START → Node1 → Node2 → Node3[FAIL]
               → Node5(fix) → Node2 → Node3[PASS] → Node4[APPROVED] → END

Path C — Review rejection, improved on first retry:
  START → Node1 → Node2 → Node3[PASS] → Node4[REJECTED]
               → Node6(improve) → Node2 → Node3[PASS] → Node4[APPROVED] → END

Path D — Test failure, fixed on second retry:
  START → Node1 → Node2 → Node3[FAIL]
               → Node5 → Node2 → Node3[FAIL]
               → Node5 → Node2 → Node3[PASS] → Node4[APPROVED] → END

Path E — Exhausted retries, still failing (graceful exit):
  START → Node1 → Node2 → Node3[FAIL]
               → Node5 → Node2 → Node3[FAIL]
               → Node5 → Node2 → Node3[FAIL]
               → Node5 → [MAX_RETRIES reached] → END (partial result)
```

### 4.2 Edge Registration in LangGraph

```python
# Fixed edges — always taken, no condition
builder.add_edge(START,            "code_generator")
builder.add_edge("code_generator", "test_generator")
builder.add_edge("test_generator", "test_runner")
builder.add_edge("code_fixer",     "test_generator")  # retry loop back
builder.add_edge("code_improver",  "test_generator")  # retry loop back

# Conditional edges — taken based on state
builder.add_conditional_edges(
    "test_runner",          # source node
    route_after_tests,      # decision function
    {                       # mapping: return value → next node
        "code_reviewer": "code_reviewer",
        "code_fixer":    "code_fixer",
        "__end__":       END,
    },
)

builder.add_conditional_edges(
    "code_reviewer",
    route_after_review,
    {
        "code_improver": "code_improver",
        "__end__":       END,
    },
)
```

---

## 5. File Structure

```
langgraph_code_review/
│
├── app.py                 ← Pipeline module (import-safe, no top-level invoke)
│   ├── AVAILABLE_MODELS   ← dict of model names → OpenRouter IDs
│   ├── DEFAULT_MODEL_KEY  ← default selection
│   ├── CodeReviewState    ← shared state TypedDict
│   ├── _require_env()     ← secure env-var reader
│   ├── _ask_llm()         ← OpenRouter API wrapper
│   ├── code_generator_node()
│   ├── test_generator_node()
│   ├── test_runner_node()
│   ├── code_reviewer_node()
│   ├── code_fixer_node()
│   ├── code_improver_node()
│   ├── route_after_tests()
│   ├── route_after_review()
│   ├── graph              ← compiled StateGraph (module-level, safe to import)
│   └── if __name__ == "__main__": graph.invoke(...)
│
├── streamlit_app.py       ← UI only; imports from app.py
│   ├── Sidebar            ← model selector, templates, pipeline diagram
│   ├── Input area         ← text_area + run/clear buttons
│   ├── Live orchestration ← per-node cards updating in real time
│   ├── Results tabs       ← Code / Tests / Test Results / Review / Fix History
│   └── Run history        ← compact log of all session runs
│
├── .env                   ← OPENROUTER_API_KEY (never committed)
├── .env.example           ← template to copy
├── .gitignore             ← excludes .env, __pycache__, venv
└── requirements.txt       ← langgraph openai python-dotenv streamlit pytest
```

---

## 6. Local Setup & Running

### Prerequisites

- Python 3.11+
- An OpenRouter account and API key — free tier at https://openrouter.ai/keys

### Installation

```bash
# 1. Clone or download the project
cd langgraph_code_review

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Edit .env → OPENROUTER_API_KEY=sk-or-v1-...
```

### Running

```bash
# ── Option A: Streamlit UI (recommended) ────────────────────────────
streamlit run streamlit_app.py
# Opens http://localhost:8501

# ── Option B: CLI only ───────────────────────────────────────────────
python app.py
# Runs with the hard-coded demo prompt in __main__

# ── Verify environment ───────────────────────────────────────────────
python -c "from app import graph, AVAILABLE_MODELS; print('✅ OK')"
```

### Common Mistakes

| Mistake | Error | Fix |
|---------|-------|-----|
| `streamlit run app.py` | Graph runs on page load | Use `streamlit run streamlit_app.py` |
| Missing `.env` | `EnvironmentError: OPENROUTER_API_KEY is not set` | `cp .env.example .env` and add key |
| Wrong model ID | `404 No endpoints found` | Check https://openrouter.ai/models?q=free |
| Python 3.9 | `TypeError: dict[str, str]` (builtin generics) | Upgrade to Python 3.11+ |

---

## 7. Scaling to Production

### 7.1 What Changes When You Scale

The current implementation is synchronous and single-process. A single `graph.invoke()` call blocks until the full pipeline completes (45–120 seconds). For production:

| Current | Production |
|---------|------------|
| `graph.invoke()` (synchronous, blocking) | Background job queue (Celery + Redis) |
| In-memory session state | PostgreSQL + Redis |
| Streamlit as full app | FastAPI backend + separate frontend |
| Single process | Horizontal pod autoscaling |
| No monitoring | LangSmith + Prometheus + Grafana |
| No caching | Semantic result cache |

### 7.2 Step-by-Step Production Migration

#### Step 1 — Extract Pipeline into an API Service

```python
# api/routes.py
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid, redis, json
from app import graph, CodeReviewState

app  = FastAPI()
rdb  = redis.Redis(host="redis", decode_responses=True)

@app.post("/run-pipeline")
async def run_pipeline(request: PipelineRequest, bg: BackgroundTasks):
    run_id = str(uuid.uuid4())
    rdb.set(f"status:{run_id}", "queued")
    bg.add_task(_execute_pipeline, run_id, request.feature_request, request.model_key)
    return {"run_id": run_id}

@app.get("/status/{run_id}")
async def get_status(run_id: str):
    status = rdb.get(f"status:{run_id}")
    return {"run_id": run_id, "status": status}

@app.get("/result/{run_id}")
async def get_result(run_id: str):
    result = rdb.get(f"result:{run_id}")
    return json.loads(result) if result else {"error": "not found"}

async def _execute_pipeline(run_id: str, feature_request: str, model_key: str):
    rdb.set(f"status:{run_id}", "running")
    initial_state: CodeReviewState = {
        "feature_request": feature_request,
        "model_key":       model_key,
        "code_content":    "", "test_content": "", "test_results": "",
        "tests_passed":    False, "review_feedback": "", "is_approved": False,
        "retry_count":     0, "fix_history": [],
    }
    result = graph.invoke(initial_state)
    rdb.set(f"result:{run_id}", json.dumps(result))
    rdb.set(f"status:{run_id}", "complete")
```

#### Step 2 — Add a Job Queue for Long-Running Tasks

```python
# workers/tasks.py
from celery import Celery
from app import graph, CodeReviewState

celery_app = Celery("pipeline", broker="redis://redis:6379/0")

@celery_app.task(bind=True, max_retries=0)  # LangGraph handles its own retries
def run_pipeline_task(self, run_id: str, feature_request: str, model_key: str):
    initial_state: CodeReviewState = {
        "feature_request": feature_request,
        "model_key":       model_key,
        "code_content":    "", "test_content": "", "test_results": "",
        "tests_passed":    False, "review_feedback": "", "is_approved": False,
        "retry_count":     0, "fix_history": [],
    }
    result = graph.invoke(initial_state)
    # Store to DB
    save_run_result(run_id, result)
    return result
```

#### Step 3 — Persist State to PostgreSQL

```python
# db/models.py
from sqlalchemy import Column, String, Boolean, Integer, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id               = Column(String, primary_key=True)   # UUID
    feature_request  = Column(String, nullable=False)
    model_key        = Column(String, nullable=False)
    code_content     = Column(String)
    test_content     = Column(String)
    test_results     = Column(String)
    tests_passed     = Column(Boolean, default=False)
    review_feedback  = Column(String)
    is_approved      = Column(Boolean, default=False)
    retry_count      = Column(Integer, default=0)
    fix_history      = Column(JSON, default=list)
    status           = Column(String, default="queued")   # queued/running/complete/failed
    created_at       = Column(DateTime, default=datetime.utcnow)
    completed_at     = Column(DateTime)
    latency_ms       = Column(Integer)   # total wall-clock time
```

#### Step 4 — Docker Compose for Local Production Testing

```yaml
# docker-compose.yml
version: "3.9"
services:
  api:
    build: .
    command: uvicorn api.routes:app --host 0.0.0.0 --port 8000
    env_file: .env
    depends_on: [redis, postgres]
    ports: ["8000:8000"]

  worker:
    build: .
    command: celery -A workers.tasks worker --concurrency=4
    env_file: .env
    depends_on: [redis, postgres]

  streamlit:
    build: .
    command: streamlit run streamlit_app.py --server.port 8501
    env_file: .env
    ports: ["8501:8501"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: pipeline
      POSTGRES_USER: pipeline
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data

  prometheus:
    image: prom/prometheus
    volumes: ["./prometheus.yml:/etc/prometheus/prometheus.yml"]
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]

volumes:
  pgdata:
```

#### Step 5 — Kubernetes for Auto-Scaling

```yaml
# k8s/worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pipeline-worker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pipeline-worker
  template:
    spec:
      containers:
        - name: worker
          image: your-registry/pipeline:latest
          command: ["celery", "-A", "workers.tasks", "worker", "--concurrency=4"]
          resources:
            requests: {cpu: "500m", memory: "1Gi"}
            limits:   {cpu: "2000m", memory: "4Gi"}
          env:
            - name: OPENROUTER_API_KEY
              valueFrom:
                secretKeyRef:
                  name: pipeline-secrets
                  key: openrouter-api-key
---
# KEDA autoscaler — scale workers based on Redis queue depth
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: pipeline-worker-scaler
spec:
  scaleTargetRef:
    name: pipeline-worker
  minReplicaCount: 1
  maxReplicaCount: 20
  triggers:
    - type: redis
      metadata:
        address: redis:6379
        listName: celery
        listLength: "5"  # 1 new worker per 5 queued jobs
```

---

## 8. Prompt Version Management

### 8.1 The Problem

Prompts are code. Changing a prompt without tracking it means:
- You cannot roll back a regression
- You cannot A/B test different phrasings
- You cannot audit why a run produced a certain output
- Your pipeline output silently changes between runs

### 8.2 Current Implementation (Development)

All prompts are currently inline strings in `app.py`. This is intentional for simplicity — easy to read, easy to understand the flow. For production, extract them.

### 8.3 Prompt Registry Pattern

```python
# prompts/registry.py
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class Prompt:
    """Immutable versioned prompt."""
    name:       str
    version:    str
    system:     str
    created_at: str
    changelog:  str

# ── All prompts registered here — never edit in place, always add new version ──

CODE_GENERATOR_V1 = Prompt(
    name       = "code_generator",
    version    = "1.0.0",
    system     = (
        "You are a senior Python engineer. "
        "Return ONLY raw Python code — no markdown, no explanations. "
        "Include type hints, a Google-style docstring, and input validation."
    ),
    created_at = "2025-05-01",
    changelog  = "Initial version",
)

CODE_GENERATOR_V2 = Prompt(
    name       = "code_generator",
    version    = "2.0.0",
    system     = (
        "You are a senior Python engineer specialising in clean, testable code. "
        "Return ONLY raw Python code — no markdown, no explanations. "
        "Rules: (1) Full type hints on every function. "
        "(2) Google-style docstring with Args, Returns, Raises. "
        "(3) Validate all inputs; raise ValueError with descriptive messages. "
        "(4) Use pathlib, not os.path. (5) No global mutable state."
    ),
    created_at = "2025-05-15",
    changelog  = "Added explicit sub-rules; improved type-hint instruction",
)

TEST_GENERATOR_V1 = Prompt(
    name       = "test_generator",
    version    = "1.0.0",
    system     = (
        "You are an expert Python test engineer. "
        "Return ONLY raw pytest code. "
        "Import with: from source import <name>  "
        "Cover normal, edge, and exception cases. "
        "Never assume rounding or formatting not present in the source."
    ),
    created_at = "2025-05-01",
    changelog  = "Initial version",
)

CODE_REVIEWER_V1 = Prompt(
    name       = "code_reviewer",
    version    = "1.0.0",
    system     = (
        "You are a senior code reviewer. "
        "Review on: correctness, quality, type hints, error handling, security. "
        "End with EXACTLY one of:\n"
        "  DECISION: APPROVED\n"
        "  DECISION: REJECTED"
    ),
    created_at = "2025-05-01",
    changelog  = "Initial version",
)

# ── Active versions (change only this dict when promoting a new version) ─────

ACTIVE_PROMPTS: dict[str, Prompt] = {
    "code_generator": CODE_GENERATOR_V2,   # promoted 2025-05-15
    "test_generator": TEST_GENERATOR_V1,
    "code_reviewer":  CODE_REVIEWER_V1,
    "code_fixer":     CODE_GENERATOR_V1,   # reuses base generator prompt
    "code_improver":  CODE_GENERATOR_V1,
}

def get_prompt(name: str) -> Prompt:
    """Get the active prompt by name."""
    if name not in ACTIVE_PROMPTS:
        raise KeyError(f"No active prompt registered for '{name}'")
    return ACTIVE_PROMPTS[name]
```

### 8.4 Using the Registry in Nodes

```python
# In app.py
from prompts.registry import get_prompt

def code_generator_node(state: CodeReviewState) -> dict:
    prompt = get_prompt("code_generator")
    print(f"  [PROMPT]  {prompt.name} v{prompt.version}")

    code = _ask_llm(
        system    = prompt.system,
        user      = f"Write Python code for:\n\n{state['feature_request']}",
        model_key = state["model_key"],
    )
    return {"code_content": code, "prompt_versions": {
        **state.get("prompt_versions", {}),
        "code_generator": prompt.version,
    }}
```

### 8.5 Storing Prompt Versions with Each Run

```python
# Add to CodeReviewState
class CodeReviewState(TypedDict):
    # ... existing fields ...
    prompt_versions: dict[str, str]   # {"code_generator": "2.0.0", ...}
```

This makes every stored run auditable — you can always replay it with the same prompt versions.

### 8.6 A/B Testing Prompts

```python
# prompts/ab_test.py
import random

def get_prompt_for_ab_test(name: str, user_id: str) -> Prompt:
    """Route 50% of users to variant B for testing."""
    # Deterministic per user — same user always sees same variant
    bucket = int(user_id[-1], 16) % 2  # 0 or 1
    
    variants = {
        "code_generator": [CODE_GENERATOR_V1, CODE_GENERATOR_V2]
    }
    
    if name in variants:
        return variants[name][bucket]
    return get_prompt(name)
```

---

## 9. Latency Optimization

### 9.1 Understanding Where Time Goes

A typical pipeline run takes **45–120 seconds**. Here is the breakdown:

```
code_generator_node   ~15s   (1 LLM call, ~1200 token output)
test_generator_node   ~12s   (1 LLM call, ~1500 token output)
test_runner_node       ~3s   (subprocess, no LLM)
code_reviewer_node    ~15s   (1 LLM call, ~2000 token output)
─────────────────────────────
Happy path total      ~45s

Per retry adds:       ~30s   (code_fixer + test_generator + test_runner)
Per review retry:     ~42s   (code_improver + test_generator + test_runner + code_reviewer)
```

### 9.2 Caching with Semantic Hashing

```python
# cache/semantic_cache.py
import hashlib, json, redis

rdb = redis.Redis(host="redis", decode_responses=True)
CACHE_TTL = 3600 * 24  # 24 hours

def _make_cache_key(feature_request: str, model_key: str, prompt_version: str) -> str:
    """Deterministic key — same inputs always produce same key."""
    payload = json.dumps({
        "feature_request": feature_request.strip().lower(),
        "model_key":       model_key,
        "prompt_version":  prompt_version,
    }, sort_keys=True)
    return "cache:" + hashlib.sha256(payload.encode()).hexdigest()

def get_cached_code(feature_request: str, model_key: str, prompt_version: str):
    key = _make_cache_key(feature_request, model_key, prompt_version)
    cached = rdb.get(key)
    return json.loads(cached) if cached else None

def cache_code(feature_request: str, model_key: str, prompt_version: str, result: dict):
    key = _make_cache_key(feature_request, model_key, prompt_version)
    rdb.setex(key, CACHE_TTL, json.dumps(result))

# Usage in code_generator_node:
def code_generator_node(state: CodeReviewState) -> dict:
    prompt  = get_prompt("code_generator")
    cached  = get_cached_code(state["feature_request"], state["model_key"], prompt.version)
    if cached:
        print("  [CACHE HIT]  Returning cached code")
        return {"code_content": cached["code_content"]}

    code = _ask_llm(...)
    cache_code(state["feature_request"], state["model_key"], prompt.version,
               {"code_content": code})
    return {"code_content": code}
```

### 9.3 Streaming Responses

```python
# For user-facing nodes, stream token-by-token instead of waiting for completion

def _ask_llm_streaming(system: str, user: str, model_key: str, on_token=None) -> str:
    """Stream response; call on_token(text) for each chunk."""
    client   = OpenAI(api_key=_require_env("OPENROUTER_API_KEY"),
                      base_url="https://openrouter.ai/api/v1")
    model_id = AVAILABLE_MODELS[model_key]

    stream  = client.chat.completions.create(
        model=model_id, max_tokens=4096, stream=True,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
    )

    full_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_text += delta
        if on_token:
            on_token(delta)   # e.g., update a Streamlit placeholder live

    return full_text.strip()
```

### 9.4 Parallel Node Execution

Nodes 1 and 2 are sequential by design, but in more complex pipelines, independent nodes can run in parallel using LangGraph's `Send` API:

```python
# If you had multiple independent code generators to compare:
from langgraph.types import Send

def fan_out_to_generators(state: CodeReviewState):
    """Run code generation with 2 models in parallel, pick the better one."""
    return [
        Send("code_generator_a", {**state, "model_key": "GPT-OSS 120B"}),
        Send("code_generator_b", {**state, "model_key": "DeepSeek R1"}),
    ]
```

### 9.5 Latency Tracking Decorator

```python
# monitoring/latency.py
import time, functools
from prometheus_client import Histogram

NODE_LATENCY = Histogram(
    "pipeline_node_latency_seconds",
    "Latency per node",
    labelnames=["node_name", "model_key"],
    buckets=[1, 5, 10, 20, 30, 60, 120],
)

def track_latency(node_name: str):
    """Decorator that records node execution time to Prometheus."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(state: CodeReviewState, *args, **kwargs):
            start = time.perf_counter()
            result = func(state, *args, **kwargs)
            elapsed = time.perf_counter() - start
            NODE_LATENCY.labels(
                node_name = node_name,
                model_key = state.get("model_key", "unknown"),
            ).observe(elapsed)
            print(f"  [LATENCY]  {node_name}: {elapsed:.2f}s")
            return result
        return wrapper
    return decorator

# Usage:
@track_latency("code_generator")
def code_generator_node(state: CodeReviewState) -> dict:
    ...
```

### 9.6 Latency Budget per Request

```python
# Enforce a per-request timeout across the whole pipeline
import signal

class TimeoutError(Exception): pass

def _timeout_handler(signum, frame):
    raise TimeoutError("Pipeline exceeded 3-minute budget")

def run_with_timeout(feature_request: str, model_key: str, timeout_sec: int = 180):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        return graph.invoke(create_initial_state(feature_request, model_key))
    except TimeoutError:
        return {"error": "Pipeline timed out", "is_approved": False}
    finally:
        signal.alarm(0)   # cancel alarm
```

---

## 10. Observability & Monitoring

### 10.1 LangSmith Integration (LLM Tracing)

LangSmith traces every LLM call — inputs, outputs, latency, token usage — with zero code changes:

```bash
# Add to .env
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=code-review-pipeline
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

With these set, every `graph.invoke()` automatically sends traces to LangSmith. You get:
- Full prompt + response for every node
- Token usage and cost per call
- Latency waterfall across all nodes
- Retry paths visualised as tree branches

### 10.2 Structured Logging

```python
# logging_config.py
import logging, json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp":  datetime.utcnow().isoformat(),
            "level":      record.levelname,
            "logger":     record.name,
            "message":    record.getMessage(),
            "run_id":     getattr(record, "run_id", None),
            "node":       getattr(record, "node", None),
            "model_key":  getattr(record, "model_key", None),
            "latency_ms": getattr(record, "latency_ms", None),
        })

def setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logging.basicConfig(handlers=[handler], level=logging.INFO)
```

### 10.3 Key Metrics to Track

| Metric | Type | Label | Why |
|--------|------|-------|-----|
| `pipeline_runs_total` | Counter | `status`, `model_key` | Overall throughput |
| `pipeline_node_latency_seconds` | Histogram | `node_name`, `model_key` | Per-node speed |
| `pipeline_retry_count` | Histogram | `reason` (test/review) | Quality signal |
| `pipeline_approval_rate` | Gauge | `model_key` | Model quality comparison |
| `llm_tokens_used_total` | Counter | `node_name`, `model_key` | Cost tracking |
| `pipeline_queue_depth` | Gauge | — | Capacity planning |

### 10.4 Grafana Dashboard Panels

```
Row 1: Health
  - Total runs today
  - Approval rate (%)
  - Average total latency (p50/p95)
  - Active retries in-flight

Row 2: Latency Waterfall
  - Per-node p50 / p95 bar chart
  - Retry frequency per stage
  - End-to-end latency over time

Row 3: Model Comparison
  - Approval rate per model
  - Average retries per model
  - Token usage per model

Row 4: Errors
  - Failed runs by reason
  - Retry exhaustion rate
  - API error rate from OpenRouter
```

---

## 11. Security Best Practices

### 11.1 API Key Security — What This Project Does

```python
# ✅ Key is read fresh on every call — not stored in any variable
def _ask_llm(system, user, model_key):
    api_key = _require_env("OPENROUTER_API_KEY")   # reads os.environ each time
    client  = OpenAI(api_key=api_key, ...)          # client created locally, goes out of scope
    ...

# ✅ Key loaded from .env, never hard-coded
load_dotenv()

# ✅ .env excluded from git
# .gitignore:  .env
```

### 11.2 Production Secrets Management

```bash
# Development: .env file
OPENROUTER_API_KEY=sk-or-v1-...

# Staging/Production: Use a secrets manager — NEVER .env files in production
# AWS Secrets Manager:
aws secretsmanager create-secret \
  --name prod/pipeline/openrouter \
  --secret-string '{"OPENROUTER_API_KEY":"sk-or-v1-..."}'

# Kubernetes: Use Secrets, not ConfigMaps
kubectl create secret generic pipeline-secrets \
  --from-literal=openrouter-api-key=sk-or-v1-...
```

### 11.3 Code Execution Safety

The `test_runner_node` runs user-generated code (from an LLM). Safety measures:

```python
# Current implementation:
# ✅ Runs in a temporary directory (auto-deleted)
# ✅ 60-second timeout prevents infinite loops
# ✅ Only captures stdout/stderr — no file system access outside tmpdir

# Production hardening — add these:
import resource, os

def _safe_subprocess(cmd, cwd, timeout=60):
    """Run subprocess with resource limits."""
    def limit_resources():
        resource.setrlimit(resource.RLIMIT_AS,   (512*1024*1024, 512*1024*1024))  # 512MB RAM
        resource.setrlimit(resource.RLIMIT_CPU,  (30, 30))                          # 30s CPU
        resource.setrlimit(resource.RLIMIT_FSIZE, (10*1024*1024, 10*1024*1024))   # 10MB disk

    return subprocess.run(
        cmd,
        capture_output=True, text=True, cwd=cwd, timeout=timeout,
        preexec_fn=limit_resources,   # applied before subprocess starts
        user="nobody",                # run as unprivileged user (Linux)
    )
```

For maximum isolation in production, run the test subprocess inside a Docker container or a gVisor sandbox.

### 11.4 Input Validation

```python
# Validate feature_request before entering the pipeline
MAX_REQUEST_LENGTH = 2000

def validate_feature_request(text: str) -> str:
    if not text or not text.strip():
        raise ValueError("Feature request cannot be empty")
    if len(text) > MAX_REQUEST_LENGTH:
        raise ValueError(f"Feature request exceeds {MAX_REQUEST_LENGTH} characters")
    # Reject obvious prompt injection attempts
    forbidden = ["ignore all previous instructions", "you are now", "jailbreak"]
    if any(phrase in text.lower() for phrase in forbidden):
        raise ValueError("Feature request contains disallowed content")
    return text.strip()
```

---

## 12. Testing Strategy

### 12.1 Unit Testing Each Node

```python
# tests/test_nodes.py
import pytest
from unittest.mock import patch, MagicMock
from app import (
    CodeReviewState, code_generator_node, test_runner_node,
    route_after_tests, route_after_review, MAX_RETRIES,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def base_state() -> CodeReviewState:
    return {
        "feature_request":  "Create an add function",
        "model_key":        "GPT-OSS 120B",
        "code_content":     "",
        "test_content":     "",
        "test_results":     "",
        "tests_passed":     False,
        "review_feedback":  "",
        "is_approved":      False,
        "retry_count":      0,
        "fix_history":      [],
    }

# ── Router unit tests (no LLM needed) ────────────────────────────────────────

def test_route_after_tests_passes(base_state):
    base_state["tests_passed"] = True
    assert route_after_tests(base_state) == "code_reviewer"

def test_route_after_tests_fails_with_retries(base_state):
    base_state["tests_passed"] = False
    base_state["retry_count"]  = 0
    assert route_after_tests(base_state) == "code_fixer"

def test_route_after_tests_exhausted(base_state):
    base_state["tests_passed"] = False
    base_state["retry_count"]  = MAX_RETRIES
    assert route_after_tests(base_state) == "__end__"

def test_route_after_review_approved(base_state):
    base_state["is_approved"] = True
    assert route_after_review(base_state) == "__end__"

def test_route_after_review_rejected_with_retries(base_state):
    base_state["is_approved"]  = False
    base_state["retry_count"]  = 0
    assert route_after_review(base_state) == "code_improver"

# ── Node tests (mock LLM) ─────────────────────────────────────────────────────

@patch("app._ask_llm", return_value="def add(a: int, b: int) -> int:\n    return a + b")
def test_code_generator_returns_code(mock_llm, base_state):
    result = code_generator_node(base_state)
    assert "code_content" in result
    assert "def add" in result["code_content"]
    mock_llm.assert_called_once()

def test_test_runner_with_valid_code(base_state):
    base_state["code_content"]  = "def add(a, b):\n    return a + b\n"
    base_state["test_content"]  = (
        "from source import add\n"
        "def test_add(): assert add(1, 2) == 3\n"
    )
    result = test_runner_node(base_state)
    assert result["tests_passed"] is True
    assert "PASSED" in result["test_results"]

def test_test_runner_with_failing_code(base_state):
    base_state["code_content"]  = "def add(a, b):\n    return a - b\n"  # bug
    base_state["test_content"]  = (
        "from source import add\n"
        "def test_add(): assert add(1, 2) == 3\n"
    )
    result = test_runner_node(base_state)
    assert result["tests_passed"] is False
    assert "FAILED" in result["test_results"]
```

### 12.2 Integration Test — Full Graph

```python
# tests/test_integration.py
@patch("app._ask_llm")
def test_full_pipeline_happy_path(mock_llm):
    """Pipeline completes without retries when code and review are correct."""
    mock_llm.side_effect = [
        # Node 1: code
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        # Node 2: tests
        "from source import add\ndef test_add():\n    assert add(1,2)==3\n",
        # Node 4: review
        "Code is excellent. DECISION: APPROVED",
    ]

    result = graph.invoke({
        "feature_request": "add two integers",
        "model_key":       "GPT-OSS 120B",
        "code_content":    "", "test_content":    "", "test_results": "",
        "tests_passed":    False, "review_feedback": "", "is_approved": False,
        "retry_count":     0,  "fix_history": [],
    })

    assert result["is_approved"] is True
    assert result["tests_passed"] is True
    assert result["retry_count"] == 0
    assert mock_llm.call_count == 3
```

---

## 13. Cost Management

### 13.1 Token Cost Estimation

All models used are currently free-tier on OpenRouter. For paid models, estimate costs:

```python
# monitoring/cost.py
COST_PER_1K_TOKENS = {
    "openai/gpt-oss-120b:free":              0.000,  # free
    "google/gemini-2.0-flash-exp:free":      0.000,  # free
    "meta-llama/llama-4-maverick:free":      0.000,  # free
    "deepseek/deepseek-r1:free":             0.000,  # free
    # Paid options for reference:
    "openai/gpt-4o":                         0.005,  # per 1K input tokens
    "anthropic/claude-sonnet-4-6":           0.003,
}

def estimate_run_cost(model_key: str, token_counts: dict) -> float:
    model_id = AVAILABLE_MODELS.get(model_key, "")
    rate     = COST_PER_1K_TOKENS.get(model_id, 0.0)
    total_k  = sum(token_counts.values()) / 1000
    return total_k * rate
```

### 13.2 Cost Guardrails

```python
MAX_TOKENS_PER_RUN = 50_000   # Hard cap across all nodes in one run

class TokenBudgetExceeded(Exception): pass

def _ask_llm_with_budget(system, user, model_key, state):
    used_so_far = state.get("total_tokens_used", 0)
    if used_so_far > MAX_TOKENS_PER_RUN:
        raise TokenBudgetExceeded(f"Run exceeded {MAX_TOKENS_PER_RUN} token budget")

    response = client.chat.completions.create(...)
    tokens   = response.usage.total_tokens
    return response.choices[0].message.content, tokens
```

---

## 14. Troubleshooting

### 14.1 Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `404 No endpoints found for <model>` | Model ID deprecated or misspelled | Check https://openrouter.ai/models?q=free |
| Pipeline runs at page load | `streamlit run app.py` used | Use `streamlit run streamlit_app.py` |
| `EnvironmentError: OPENROUTER_API_KEY not set` | No `.env` file or empty key | `cp .env.example .env` and add key |
| Tests always fail on float comparison | LLM generates wrong expected values | System prompt instructs: "match implementation exactly" |
| Infinite retry (rare) | `route_after_tests` always returns `code_fixer` | Check `MAX_RETRIES` value and `retry_count` increment |
| `ModuleNotFoundError: No module named 'langgraph'` | venv not activated | `source venv/bin/activate` |
| Slow performance (>3 min) | Model overloaded or long prompts | Switch to a smaller model or add caching |
| `streamlit.errors.StreamlitAPIException` | `st.set_page_config` not first call | Move `set_page_config` to top of file |

### 14.2 Debugging a Specific Run

```python
# Enable verbose LangGraph output:
import logging
logging.getLogger("langgraph").setLevel(logging.DEBUG)

# Or, inspect state after each node manually:
state = create_initial_state("Create an add function")
state.update(code_generator_node(state))
print("After Node 1:", state["code_content"][:100])

state.update(test_generator_node(state))
print("After Node 2:", state["test_content"][:100])
# ... and so on
```

### 14.3 Checking the Graph Structure

```python
# Visualise the graph as a Mermaid diagram
from app import graph
print(graph.get_graph().draw_mermaid())

# Output:
# graph TD;
#   __start__ --> code_generator;
#   code_generator --> test_generator;
#   test_generator --> test_runner;
#   test_runner -.-> code_reviewer;
#   test_runner -.-> code_fixer;
#   test_runner -.-> __end__;
#   code_reviewer -.-> code_improver;
#   code_reviewer -.-> __end__;
#   code_fixer --> test_generator;
#   code_improver --> test_generator;
```

---

## Summary — Quick Reference

```
┌──────────────────────────────────────────────────────────────────┐
│                    QUICK REFERENCE                                │
├──────────────────────────────────────────────────────────────────┤
│  Run UI           streamlit run streamlit_app.py                 │
│  Run CLI          python app.py                                  │
│  Run Tests        pytest tests/ -v                               │
│  Add model        Edit AVAILABLE_MODELS in app.py                │
│  Change retries   Edit MAX_RETRIES in app.py                     │
│  Change prompts   Edit system= strings in each node function     │
│  API Key          OPENROUTER_API_KEY in .env                     │
│  Free models      https://openrouter.ai/models?q=free            │
│  LLM Traces       Set LANGCHAIN_API_KEY + LANGCHAIN_TRACING_V2   │
├──────────────────────────────────────────────────────────────────┤
│                    NODE RESPONSIBILITY MAP                        │
├──────────────────────────────────────────────────────────────────┤
│  Node 1  code_generator    feature_request → code_content        │
│  Node 2  test_generator    code_content    → test_content        │
│  Node 3  test_runner       code+tests      → test_results (real) │
│  Node 4  code_reviewer     code+results    → review + decision   │
│  Node 5  code_fixer        code+failures   → code (fixed)        │
│  Node 6  code_improver     code+review     → code (improved)     │
├──────────────────────────────────────────────────────────────────┤
│                    CONDITIONAL EDGES                              │
├──────────────────────────────────────────────────────────────────┤
│  test_runner  → PASS          → code_reviewer                    │
│  test_runner  → FAIL+retries  → code_fixer                       │
│  test_runner  → FAIL+max      → END (graceful)                   │
│  code_reviewer → APPROVED     → END (success)                    │
│  code_reviewer → REJECTED+ret → code_improver                    │
│  code_reviewer → REJECTED+max → END (graceful)                   │
└──────────────────────────────────────────────────────────────────┘
```