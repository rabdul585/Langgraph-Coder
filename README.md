# ⚙️ LangGraph Code Review Pipeline

A **self-healing, multi-agent AI system** that takes a plain-English feature request and automatically generates, tests, and reviews production-quality Python code — retrying and learning from its own mistakes until the code is approved.

Built with **LangGraph** · **OpenRouter** · **Streamlit** · **Python 3.11+**

---

## ✨ What Makes This Different

Most AI code generators stop when they fail. This one **doesn't**.

When tests fail, the agent reads the exact error output and rewrites the code to fix it. When a reviewer rejects it, the agent reads every concern and improves the code — then re-tests and re-reviews. This loop continues automatically until the code passes or the retry limit is reached.

```
You type:   "Create a function to validate email addresses"

Pipeline:
  🤖 Generates code  →  🧪 Generates tests  →  ▶️ Runs tests
                                                      │
                              ┌───────── FAIL ────────┘
                              │
                         🔧 Fixes code (reads error output)
                              │
                         🧪 Regenerates tests  →  ▶️ Re-runs tests
                                                      │
                              ┌──────────── PASS ─────┘
                              │
                         👀 Reviews code quality
                              │
                    REJECTED ─┤
                              │
                         ✨ Improves code (reads review feedback)
                              │
                         (loops back to testing)
                              │
                    APPROVED ─┘

You get:    Production-ready Python with full test suite + review report
```

---

## 🏗️ Architecture

### Pipeline Flow

```
START
  ↓
🤖 Node 1 — Code Generator
   reads  ← feature_request, model_key
   writes → code_content
  ↓
🧪 Node 2 — Test Generator
   reads  ← code_content, feature_request
   writes → test_content
  ↓
▶️  Node 3 — Test Runner  (real pytest subprocess — no LLM)
   reads  ← code_content, test_content
   writes → test_results, tests_passed
  ↓
  ├─ PASS ──────────────────────────────────────────┐
  │                                                 ↓
  └─ FAIL ──→ 🔧 Node 5 — Code Fixer        👀 Node 4 — Code Reviewer
               reads  ← code_content              reads  ← code_content
                         test_results                       test_results
               writes → code_content (fixed)      writes → review_feedback
                         retry_count + 1                    is_approved
                         fix_history               ↓
               ↓         ↓                    ├─ APPROVED ──→ END ✅
               └─────────┘                    │
               (back to Node 2)               └─ REJECTED ──→ ✨ Node 6 — Code Improver
                                                              reads  ← code_content
                                                                        review_feedback
                                                              writes → code_content (improved)
                                                                        retry_count + 1
                                                                        fix_history
                                                              ↓
                                                              (back to Node 2)
```

### Two Files, Clean Separation

| File | Role |
|------|------|
| `app.py` | Pure pipeline logic. Safe to import. `graph.invoke()` only inside `if __name__ == "__main__"` |
| `streamlit_app.py` | UI only. Imports `graph` and node functions from `app.py`. Never runs the pipeline at import time |

### The 6 Nodes

| # | Node | LLM? | Reads from State | Writes to State |
|---|------|-------|-----------------|-----------------|
| 1 | `code_generator_node` | ✅ | `feature_request`, `model_key` | `code_content` |
| 2 | `test_generator_node` | ✅ | `code_content`, `feature_request` | `test_content` |
| 3 | `test_runner_node` | ❌ (subprocess) | `code_content`, `test_content` | `test_results`, `tests_passed` |
| 4 | `code_reviewer_node` | ✅ | `code_content`, `test_results`, `tests_passed` | `review_feedback`, `is_approved` |
| 5 | `code_fixer_node` | ✅ | `code_content`, `test_results` | `code_content`, `retry_count`, `fix_history` |
| 6 | `code_improver_node` | ✅ | `code_content`, `review_feedback` | `code_content`, `retry_count`, `fix_history` |

### Conditional Edges (The Retry Logic)

```python
# After test_runner — decides next step based on test outcome
route_after_tests(state):
    tests passed         →  code_reviewer
    tests failed + retry →  code_fixer        # learns from error, rewrites
    tests failed + max   →  END (graceful)    # cap: MAX_RETRIES = 3

# After code_reviewer — decides next step based on approval
route_after_review(state):
    is_approved          →  END               # success
    rejected + retry     →  code_improver     # learns from feedback, rewrites
    rejected + max       →  END (graceful)
```

### Shared State (The Notepad)

Every node reads from and writes to one `TypedDict`. LangGraph merges updates automatically — nodes never re-pass fields they didn't change.

```python
class CodeReviewState(TypedDict):
    feature_request:  str    # INPUT — what to build
    model_key:        str    # INPUT — which model to use
    code_content:     str    # Node 1/5/6 — generated Python code
    test_content:     str    # Node 2     — pytest test suite
    test_results:     str    # Node 3     — pytest stdout/stderr
    tests_passed:     bool   # Node 3     — True if all tests pass
    review_feedback:  str    # Node 4     — review prose from LLM
    is_approved:      bool   # Node 4     — True = approved for production
    retry_count:      int    # Retry tracker (cap = MAX_RETRIES)
    fix_history:      list   # [{attempt, reason, old_code, new_code}]
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- A free OpenRouter API key → [openrouter.ai/keys](https://openrouter.ai/keys)

### 1. Install

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy the example file
cp .env.example .env

# Open .env and add your key
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 3. Run

```bash
# ── Streamlit UI (recommended) ──
streamlit run streamlit_app.py
# Opens http://localhost:8501

# ── CLI only ──
python app.py
```

> ⚠️ **Important:** Always use `streamlit run streamlit_app.py`, not `streamlit run app.py`. Running `app.py` directly as a Streamlit app causes the pipeline to execute at page load before the user enters any input.

---

## 🤖 Available Models

All models route through [OpenRouter](https://openrouter.ai) using the OpenAI-compatible API. All listed models are **free tier**.

| Display Name | OpenRouter Model ID | Strength |
|-------------|---------------------|----------|
| GPT-OSS 120B *(default)* | `openai/gpt-oss-120b:free` | Strong code generation |
| Gemini 2.0 Flash | `google/gemini-2.0-flash-exp:free` | Fast, good reasoning |
| Llama 4 Maverick | `meta-llama/llama-4-maverick:free` | Balanced, open-source |
| DeepSeek R1 | `deepseek/deepseek-r1:free` | Strong logical reasoning |

To add a model, edit `AVAILABLE_MODELS` in `app.py`:

```python
AVAILABLE_MODELS: dict[str, str] = {
    "My New Model": "provider/model-id:free",
    # ... existing models
}
```

Browse available free models at [openrouter.ai/models?q=free](https://openrouter.ai/models?q=free)

---

## 🖥️ Streamlit UI Walkthrough

### Input Area
- **Feature Request** text box — describe the Python code you want in plain English
- **▶️ Run Pipeline** button — starts the agent
- **🗑️ Clear** button — resets results

### Sidebar
- **Model selector** — pick any model from `AVAILABLE_MODELS`
- **Quick templates** — pre-built requests (Email Validator, Prime Checker, etc.)
- **Pipeline diagram** — live view of the graph structure
- **API key status** — green ✅ if loaded, red ❌ if missing

### Live Orchestration Cards

Each node gets a colour-coded status card that updates in real time:

| Card Colour | Meaning |
|-------------|---------|
| ⬜ Gray (idle) | Waiting to run |
| 🟡 Yellow (running) | Currently executing |
| 🟢 Green (ok) | Completed successfully |
| 🔴 Red (err) | Failed or rejected |
| 🟠 Orange (warn) | Completed but with caveats (e.g. tests pass after retry) |

Retry nodes (🔧 Code Fixer and ✨ Code Improver) only appear when triggered.

### Results Tabs

| Tab | Content |
|-----|---------|
| 🤖 Generated Code | Final Python code with line/function/character metrics |
| 🧪 Test Cases | Final pytest suite with test function count |
| ▶️ Test Results | Raw pytest stdout with pass/fail summary |
| 👀 Code Review | Full review text from the LLM |
| 🔧 Fix History | Side-by-side before/after for every retry (only appears if retries occurred) |

---

## 📁 File Structure

```
langgraph_code_review/
│
├── app.py                  ← Pipeline module (import-safe)
│   ├── AVAILABLE_MODELS    ← dict: display name → OpenRouter model ID
│   ├── DEFAULT_MODEL_KEY   ← default model
│   ├── MAX_RETRIES         ← retry cap (default: 3)
│   ├── CodeReviewState     ← shared TypedDict flowing through all nodes
│   ├── _require_env()      ← secure env-var reader (never stored globally)
│   ├── _ask_llm()          ← OpenRouter API wrapper (fences stripped)
│   ├── code_generator_node()
│   ├── test_generator_node()
│   ├── test_runner_node()  ← real pytest subprocess, no LLM
│   ├── code_reviewer_node()
│   ├── code_fixer_node()   ← reads test failures, rewrites code
│   ├── code_improver_node()← reads review feedback, rewrites code
│   ├── route_after_tests() ← conditional edge router
│   ├── route_after_review()← conditional edge router
│   └── graph               ← compiled LangGraph StateGraph
│
├── streamlit_app.py        ← UI only; imports from app.py
│
├── .env                    ← API key (git-ignored)
├── .env.example            ← Template — copy this
├── .gitignore              ← Excludes .env, __pycache__, venv/
├── requirements.txt        ← Python dependencies
└── DOCUMENTATION.md        ← Full technical documentation (1,400+ lines)
```

---

## 🔑 Security

### API Key Handling

The `OPENROUTER_API_KEY` is:
- Loaded from `.env` via `python-dotenv`
- **Read fresh from `os.environ` on every LLM call** — never stored in a module-level variable
- Never logged, never serialised into state
- Excluded from git via `.gitignore`

```python
# The key is read inside _ask_llm() on every call:
def _ask_llm(system, user, model_key):
    api_key = _require_env("OPENROUTER_API_KEY")  # reads os.environ each time
    client  = OpenAI(api_key=api_key, ...)         # local scope — gone after return
```

### Code Execution Safety

Node 3 runs generated code in a subprocess with:
- **Isolated temp directory** — auto-deleted after each run
- **60-second timeout** — prevents infinite loops
- **Captured output only** — no file system writes outside the temp directory

### Production Secrets

For production deployments, use a secrets manager instead of `.env` files:

```bash
# AWS Secrets Manager
aws secretsmanager create-secret \
  --name prod/pipeline/openrouter \
  --secret-string '{"OPENROUTER_API_KEY":"sk-or-v1-..."}'

# Kubernetes Secret
kubectl create secret generic pipeline-secrets \
  --from-literal=openrouter-api-key=sk-or-v1-...
```

---

## ⚙️ Configuration

All configuration lives in `.env`:

```bash
# Required
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional — uncomment to enable LLM tracing via LangSmith
# LANGCHAIN_API_KEY=ls__...
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_PROJECT=code-review-pipeline
```

Tunable constants in `app.py`:

```python
MAX_RETRIES = 3          # How many times to retry before giving up
DEFAULT_MODEL_KEY = "GPT-OSS 120B"  # Default model in Streamlit sidebar
```

---

## 🧪 Running Tests

```bash
# Install dev dependencies (pytest is already in requirements.txt)
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=term-missing
```

Test structure:

```
tests/
├── test_routes.py       # Unit tests for conditional edge functions (no LLM)
├── test_nodes.py        # Node tests with mocked _ask_llm
└── test_integration.py  # Full graph with mocked LLM responses
```

Example — routing tests need no LLM or API key:

```python
def test_route_passes_when_tests_pass(base_state):
    base_state["tests_passed"] = True
    assert route_after_tests(base_state) == "code_reviewer"

def test_route_fixes_when_tests_fail_with_retries_left(base_state):
    base_state["tests_passed"] = False
    base_state["retry_count"]  = 0
    assert route_after_tests(base_state) == "code_fixer"

def test_route_ends_when_retries_exhausted(base_state):
    base_state["tests_passed"] = False
    base_state["retry_count"]  = MAX_RETRIES
    assert route_after_tests(base_state) == "__end__"
```

---

## 🐛 Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Pipeline runs at page load | Used `streamlit run app.py` | Use `streamlit run streamlit_app.py` |
| `EnvironmentError: OPENROUTER_API_KEY is not set` | Missing `.env` file | `cp .env.example .env` then add key |
| `404 No endpoints found for <model>` | Model deprecated or wrong ID | Check [openrouter.ai/models?q=free](https://openrouter.ai/models?q=free) |
| Tests always fail on float values | LLM generated wrong expected values | Already handled — system prompt says "match implementation exactly" |
| `ModuleNotFoundError: No module named 'langgraph'` | venv not activated | `source venv/bin/activate` |
| `TypeError: dict[str, str]` | Python < 3.11 | Upgrade to Python 3.11+ |
| Slow runs (>2 min) | Model overloaded or rate-limited | Switch model in sidebar |

### Debug a specific node

```python
# Run nodes individually to isolate issues:
from app import code_generator_node, test_runner_node, CodeReviewState

state: CodeReviewState = {
    "feature_request": "Create an add function",
    "model_key": "GPT-OSS 120B",
    "code_content": "", "test_content": "", "test_results": "",
    "tests_passed": False, "review_feedback": "", "is_approved": False,
    "retry_count": 0, "fix_history": [],
}

# Test Node 1 alone:
state.update(code_generator_node(state))
print(state["code_content"])
```

### Visualise the graph

```python
from app import graph
print(graph.get_graph().draw_mermaid())
```

---

## 📚 Learn More

| Resource | Link |
|----------|------|
| LangGraph Documentation | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) |
| LangGraph Conditional Edges | [How-to: branching](https://langchain-ai.github.io/langgraph/how-tos/branching/) |
| OpenRouter Models | [openrouter.ai/models](https://openrouter.ai/models) |
| OpenRouter Free Models | [openrouter.ai/models?q=free](https://openrouter.ai/models?q=free) |
| Streamlit Documentation | [docs.streamlit.io](https://docs.streamlit.io) |
| LangSmith Tracing | [smith.langchain.com](https://smith.langchain.com) |
| Full Technical Docs | [`DOCUMENTATION.md`](DOCUMENTATION.md) |

---

## 🗺️ Roadmap

- [ ] **LangSmith tracing** — one-line integration, full visibility into every LLM call
- [ ] **Streaming output** — see generated code appear token-by-token
- [ ] **Prompt version registry** — versioned, auditable prompts with A/B testing
- [ ] **Result persistence** — save runs to SQLite / PostgreSQL
- [ ] **GitHub PR integration** — push approved code directly as a pull request
- [ ] **Multi-file generation** — generate modules, not just single functions
- [ ] **Custom reviewer rules** — configure what the reviewer checks per project
- [ ] **FastAPI backend** — decouple UI from pipeline for scalable deployments

---

## 📄 License

MIT — see `LICENSE` file.

---

<div align="center">

Built with ❤️ using [LangGraph](https://langchain-ai.github.io/langgraph/) · [OpenRouter](https://openrouter.ai) · [Streamlit](https://streamlit.io)

</div>
