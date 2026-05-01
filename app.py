"""
LangGraph Code Review Pipeline — app.py  (with Retry Loop)
============================================================

Pipeline Flow with Conditional Edges:

    START
      ↓
    [code_generator]   → writes code_content
      ↓
    [test_generator]   → writes test_content
      ↓
    [test_runner]      → writes test_results, tests_passed
      ↓
    <route_after_tests>          ← CONDITIONAL EDGE
      ├── tests PASSED  ──────→ [code_reviewer]
      └── tests FAILED  ──────→ [code_fixer]      ← learns from error, rewrites code
                                    ↓
                                [test_generator]   ← regenerate tests for fixed code
                                    ↓
                                [test_runner]      ← re-run tests
                                    ↓
                               (loop until MAX_RETRIES or tests pass)

    [code_reviewer]
      ↓
    <route_after_review>         ← CONDITIONAL EDGE
      ├── APPROVED  ───────────→ END
      └── REJECTED  ───────────→ [code_improver]  ← learns from review, rewrites code
                                    ↓
                                [test_generator]
                                    ↓
                                [test_runner]
                                    ↓
                                [code_reviewer]
                                    ↓
                               (loop until MAX_RETRIES or approved)

MAX_RETRIES = 3  (safety cap — prevents infinite loops)

Run directly (CLI):
    python app.py

Run as Streamlit UI:
    streamlit run streamlit_app.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# LOAD .env
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

MAX_RETRIES = 3   # maximum fix attempts before giving up


def _require_env(key: str) -> str:
    value = os.environ.get(key, "").strip()
    if not value:
        raise EnvironmentError(
            f"\n\n  ❌  '{key}' is not set.\n"
            f"  Create a .env file and add:\n"
            f"      {key}=your_key_here\n"
        )
    return value


# ─────────────────────────────────────────────────────────────────────────────
# AVAILABLE MODELS
# ─────────────────────────────────────────────────────────────────────────────

AVAILABLE_MODELS: dict[str, str] = {
    "GPT-OSS 120B":     "openai/gpt-oss-120b:free",
    "Gemini 2.0 Flash": "google/gemini-2.0-flash-exp:free",
    "Llama 4 Maverick": "meta-llama/llama-4-maverick:free",
    "DeepSeek R1":      "deepseek/deepseek-r1:free",
}

DEFAULT_MODEL_KEY = "GPT-OSS 120B"


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

class CodeReviewState(TypedDict):
    # ── Inputs ──────────────────────────────────────────────────────────────
    feature_request:  str    # plain-English description of what to build
    model_key:        str    # key from AVAILABLE_MODELS

    # ── Node outputs ─────────────────────────────────────────────────────────
    code_content:     str    # Node 1 / Node 5 — Python source code
    test_content:     str    # Node 2 — pytest test suite
    test_results:     str    # Node 3 — pytest stdout/stderr
    tests_passed:     bool   # Node 3 — True when pytest exits 0
    review_feedback:  str    # Node 4 — prose review from the LLM
    is_approved:      bool   # Node 4 — True = approved for production

    # ── Retry tracking ───────────────────────────────────────────────────────
    retry_count:      int    # total fix attempts so far (cap = MAX_RETRIES)
    fix_history:      list   # list of dicts: {attempt, reason, old_code, new_code}


# ─────────────────────────────────────────────────────────────────────────────
# LLM HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _ask_llm(system: str, user: str, model_key: str) -> str:
    """Call OpenRouter and return clean plain-text (fences stripped)."""
    api_key  = _require_env("OPENROUTER_API_KEY")
    model_id = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS[DEFAULT_MODEL_KEY])

    print(f"\n  [LLM]  {model_key!r}  →  {model_id!r}")

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    response = client.chat.completions.create(
        model      = model_id,
        max_tokens = 4096,
        messages   = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )

    text = response.choices[0].message.content.strip()

    # Strip markdown fences if the model added them
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — CODE GENERATOR
# Reads:  feature_request, model_key
# Writes: code_content
# ─────────────────────────────────────────────────────────────────────────────

def code_generator_node(state: CodeReviewState) -> dict:
    print("\n" + "=" * 60)
    print("🤖  NODE 1 — CODE GENERATOR")
    print("=" * 60)
    print(f"Reading  ← feature_request: {state['feature_request'][:80]}\n")

    code = _ask_llm(
        system=(
            "You are a senior Python engineer. "
            "Return ONLY raw Python code — no markdown, no explanations. "
            "Include type hints, a Google-style docstring, and input validation."
        ),
        user=f"Write Python code for this requirement:\n\n{state['feature_request']}",
        model_key=state["model_key"],
    )

    print("\nWriting  → code_content:\n")
    print(code)
    return {"code_content": code}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — TEST GENERATOR
# Reads:  feature_request, code_content, model_key
# Writes: test_content
# ─────────────────────────────────────────────────────────────────────────────

def test_generator_node(state: CodeReviewState) -> dict:
    print("\n" + "=" * 60)
    print("🧪  NODE 2 — TEST GENERATOR")
    print("=" * 60)
    print(f"Reading  ← code_content ({len(state['code_content'])} chars)\n")

    tests = _ask_llm(
        system=(
            "You are an expert Python test engineer. "
            "Return ONLY raw pytest code — no markdown, no explanations. "
            "Import the function with:  from source import <name>  "
            "Cover normal cases, edge cases, and expected exceptions. "
            "Make sure expected values exactly match what the implementation returns — "
            "do not assume rounding or formatting that is not in the source code."
        ),
        user=(
            f"Requirement:\n{state['feature_request']}\n\n"
            f"Source code:\n{state['code_content']}\n\n"
            "Write the full pytest test suite."
        ),
        model_key=state["model_key"],
    )

    print("\nWriting  → test_content:\n")
    print(tests)
    return {"test_content": tests}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 — TEST RUNNER   (pure subprocess — no LLM)
# Reads:  code_content, test_content
# Writes: test_results, tests_passed
# ─────────────────────────────────────────────────────────────────────────────

def test_runner_node(state: CodeReviewState) -> dict:
    print("\n" + "=" * 60)
    print("▶️   NODE 3 — TEST RUNNER")
    print("=" * 60)
    print("Reading  ← code_content + test_content")
    print("Running pytest …\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "source.py").write_text(
            state["code_content"], encoding="utf-8"
        )
        (Path(tmpdir) / "test_source.py").write_text(
            state["test_content"], encoding="utf-8"
        )
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "test_source.py", "-v", "--tb=short"],
            capture_output=True, text=True, cwd=tmpdir, timeout=60,
        )

    output = proc.stdout + proc.stderr
    passed = proc.returncode == 0

    print(output)
    print(f"Writing  → tests_passed: {passed}")
    return {"test_results": output, "tests_passed": passed}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 — CODE REVIEWER
# Reads:  code_content, test_results, tests_passed, model_key
# Writes: review_feedback, is_approved
# ─────────────────────────────────────────────────────────────────────────────

def code_reviewer_node(state: CodeReviewState) -> dict:
    print("\n" + "=" * 60)
    print("👀  NODE 4 — CODE REVIEWER")
    print("=" * 60)
    print(f"Reading  ← tests_passed: {state['tests_passed']}\n")

    status = "PASSED ✅" if state["tests_passed"] else "FAILED ❌"

    feedback = _ask_llm(
        system=(
            "You are a senior code reviewer. "
            "Review on: correctness, quality, type hints, error handling, security. "
            "End with EXACTLY one of:\n"
            "  DECISION: APPROVED\n"
            "  DECISION: REJECTED"
        ),
        user=(
            f"Source code:\n{state['code_content']}\n\n"
            f"Test results (status: {status}):\n{state['test_results']}\n\n"
            "Write the code review."
        ),
        model_key=state["model_key"],
    )

    approved = "DECISION: APPROVED" in feedback.upper()

    print("\n" + feedback)
    print(f"\nWriting  → is_approved: {approved}")
    return {"review_feedback": feedback, "is_approved": approved}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5 — CODE FIXER   (retry when tests FAIL)
# Reads:  code_content, test_results, feature_request, fix_history, model_key
# Writes: code_content, retry_count, fix_history
#
# This node is reached via a CONDITIONAL EDGE from test_runner when
# tests_passed == False.  It reads the failure output and rewrites
# the code to fix the problem, then the graph loops back to test_generator.
# ─────────────────────────────────────────────────────────────────────────────

def code_fixer_node(state: CodeReviewState) -> dict:
    attempt = state["retry_count"] + 1
    print("\n" + "=" * 60)
    print(f"🔧  NODE 5 — CODE FIXER  (attempt {attempt}/{MAX_RETRIES})")
    print("=" * 60)
    print("Reading  ← code_content + test_results (failures)\n")

    fixed_code = _ask_llm(
        system=(
            "You are a senior Python engineer fixing failing tests. "
            "You will receive the original code and the pytest failure output. "
            "Analyse EVERY failure carefully and rewrite the code to fix ALL of them. "
            "Return ONLY raw Python code — no markdown, no explanations."
        ),
        user=(
            f"Original requirement:\n{state['feature_request']}\n\n"
            f"Current code (has bugs):\n{state['code_content']}\n\n"
            f"Pytest failure output:\n{state['test_results']}\n\n"
            "Rewrite the code to fix every failing test. "
            "Keep passing tests green. Return ONLY the fixed Python code."
        ),
        model_key=state["model_key"],
    )

    history = list(state.get("fix_history", []))
    history.append({
        "attempt":   attempt,
        "reason":    "test_failure",
        "old_code":  state["code_content"][:300],
        "new_code":  fixed_code[:300],
    })

    print("\nWriting  → code_content (fixed):\n")
    print(fixed_code)
    return {
        "code_content": fixed_code,
        "retry_count":  attempt,
        "fix_history":  history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 6 — CODE IMPROVER   (retry when reviewer REJECTS)
# Reads:  code_content, review_feedback, feature_request, fix_history, model_key
# Writes: code_content, retry_count, fix_history
#
# Reached via CONDITIONAL EDGE from code_reviewer when is_approved == False.
# Reads the reviewer's specific feedback and rewrites the code to address it.
# ─────────────────────────────────────────────────────────────────────────────

def code_improver_node(state: CodeReviewState) -> dict:
    attempt = state["retry_count"] + 1
    print("\n" + "=" * 60)
    print(f"✨  NODE 6 — CODE IMPROVER  (attempt {attempt}/{MAX_RETRIES})")
    print("=" * 60)
    print("Reading  ← code_content + review_feedback (rejection reasons)\n")

    improved_code = _ask_llm(
        system=(
            "You are a senior Python engineer improving code based on a code review. "
            "Read the reviewer's feedback carefully and address EVERY concern raised. "
            "Return ONLY raw Python code — no markdown, no explanations."
        ),
        user=(
            f"Original requirement:\n{state['feature_request']}\n\n"
            f"Current code (was rejected):\n{state['code_content']}\n\n"
            f"Reviewer's feedback:\n{state['review_feedback']}\n\n"
            "Rewrite the code to fully address every concern in the review. "
            "Return ONLY the improved Python code."
        ),
        model_key=state["model_key"],
    )

    history = list(state.get("fix_history", []))
    history.append({
        "attempt":   attempt,
        "reason":    "review_rejection",
        "old_code":  state["code_content"][:300],
        "new_code":  improved_code[:300],
    })

    print("\nWriting  → code_content (improved):\n")
    print(improved_code)
    return {
        "code_content": improved_code,
        "retry_count":  attempt,
        "fix_history":  history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL EDGE FUNCTIONS
# These are called by LangGraph after specific nodes to decide the next node.
# ─────────────────────────────────────────────────────────────────────────────

def route_after_tests(state: CodeReviewState) -> Literal["code_reviewer", "code_fixer", "__end__"]:
    """
    After test_runner:
      - tests passed → send to code_reviewer
      - tests failed + retries left → send to code_fixer
      - tests failed + no retries left → end (give up)
    """
    if state["tests_passed"]:
        print("\n  ✅  Tests passed → routing to code_reviewer")
        return "code_reviewer"
    elif state["retry_count"] < MAX_RETRIES:
        print(f"\n  ❌  Tests failed → routing to code_fixer "
              f"(attempt {state['retry_count'] + 1}/{MAX_RETRIES})")
        return "code_fixer"
    else:
        print(f"\n  ⛔  Tests failed after {MAX_RETRIES} retries → ending")
        return "__end__"


def route_after_review(state: CodeReviewState) -> Literal["code_improver", "__end__"]:
    """
    After code_reviewer:
      - approved → end (success)
      - rejected + retries left → send to code_improver
      - rejected + no retries left → end (give up)
    """
    if state["is_approved"]:
        print("\n  ✅  Code approved → ending (success)")
        return "__end__"
    elif state["retry_count"] < MAX_RETRIES:
        print(f"\n  ❌  Code rejected → routing to code_improver "
              f"(attempt {state['retry_count'] + 1}/{MAX_RETRIES})")
        return "code_improver"
    else:
        print(f"\n  ⛔  Rejected after {MAX_RETRIES} retries → ending")
        return "__end__"


# ─────────────────────────────────────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────────────────────────────────────

builder = StateGraph(CodeReviewState)

# ── Register all nodes ───────────────────────────────────────────────────────
builder.add_node("code_generator", code_generator_node)   # Node 1
builder.add_node("test_generator", test_generator_node)   # Node 2
builder.add_node("test_runner",    test_runner_node)       # Node 3
builder.add_node("code_reviewer",  code_reviewer_node)    # Node 4
builder.add_node("code_fixer",     code_fixer_node)       # Node 5  (retry on test fail)
builder.add_node("code_improver",  code_improver_node)    # Node 6  (retry on rejection)

# ── Fixed edges (always taken) ───────────────────────────────────────────────
builder.add_edge(START,            "code_generator")
builder.add_edge("code_generator", "test_generator")
builder.add_edge("test_generator", "test_runner")

# After code_fixer → regenerate tests → rerun → re-evaluate
builder.add_edge("code_fixer",    "test_generator")

# After code_improver → regenerate tests → rerun → re-review
builder.add_edge("code_improver", "test_generator")

# ── Conditional edges (route based on state) ─────────────────────────────────
#
#  After test_runner:
#    tests passed  →  code_reviewer
#    tests failed  →  code_fixer   (or __end__ if retries exhausted)
#
builder.add_conditional_edges(
    "test_runner",
    route_after_tests,
    {
        "code_reviewer": "code_reviewer",
        "code_fixer":    "code_fixer",
        "__end__":       END,
    },
)

#  After code_reviewer:
#    approved  →  END
#    rejected  →  code_improver  (or __end__ if retries exhausted)
#
builder.add_conditional_edges(
    "code_reviewer",
    route_after_review,
    {
        "code_improver": "code_improver",
        "__end__":       END,
    },
)

# Compile — validates graph, returns runnable object
graph = builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT — only runs when called as:  python app.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    initial_state: CodeReviewState = {
        "feature_request": (
            "Create a Python function called `calculate_discount` that takes "
            "a price (float) and a discount percentage (float, 0–100) and "
            "returns the discounted price. Raise ValueError for invalid inputs."
        ),
        "model_key":       DEFAULT_MODEL_KEY,
        "code_content":    "",
        "test_content":    "",
        "test_results":    "",
        "tests_passed":    False,
        "review_feedback": "",
        "is_approved":     False,
        "retry_count":     0,
        "fix_history":     [],
    }

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("📊  FINAL STATE")
    print("=" * 60)
    print(f"  model_key        : {final_state['model_key']}")
    print(f"  tests_passed     : {final_state['tests_passed']}")
    print(f"  is_approved      : {final_state['is_approved']}")
    print(f"  retry_count      : {final_state['retry_count']}")
    print(f"  fix_history      : {len(final_state['fix_history'])} attempt(s)")
    print(f"  code_content     : {len(final_state['code_content'])} chars")
    print(f"  review_feedback  : {len(final_state['review_feedback'])} chars")
    print("=" * 60)