"""
Streamlit UI — Code Review Pipeline (with Retry Loop)
======================================================

Run with:
    streamlit run streamlit_app.py

Shows each node updating live, including retry attempts with reasons.
"""

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app import (  # noqa: E402
    AVAILABLE_MODELS,
    DEFAULT_MODEL_KEY,
    MAX_RETRIES,
    CodeReviewState,
    code_generator_node,
    test_generator_node,
    test_runner_node,
    code_reviewer_node,
    code_fixer_node,
    code_improver_node,
    route_after_tests,
    route_after_review,
)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Code Review Pipeline",
    page_icon="⚙️",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Node status cards ──────────────────────────────────── */
.nc {
    border: 1.5px solid #e2e8f0; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px; background: #f8fafc;
}
.nc.idle    { border-color: #cbd5e1; background: #f8fafc; }
.nc.running { border-color: #f59e0b; background: #fffbeb; }
.nc.ok      { border-color: #22c55e; background: #f0fdf4; }
.nc.warn    { border-color: #f97316; background: #fff7ed; }
.nc.err     { border-color: #ef4444; background: #fff1f2; }

.nc-title { font-weight: 700; font-size: 0.92rem; margin-bottom: 3px; }
.nc-rw    { font-size: 0.72rem; color: #64748b; font-family: monospace; }
.nc-note  { font-size: 0.78rem; color: #374151; margin-top: 6px; }

/* attempt badge */
.attempt-badge {
    display: inline-block; background: #fef3c7; color: #92400e;
    border-radius: 10px; padding: 1px 9px; font-size: 0.7rem;
    font-weight: 700; margin-left: 8px;
}
.attempt-badge.fix { background: #fee2e2; color: #991b1b; }

/* ── Result tab badges ──────────────────────────────────── */
.tb { display:inline-block; padding:2px 12px; border-radius:20px;
      font-size:0.72rem; font-weight:700; margin-bottom:6px; }
.tb1 { background:#dbeafe; color:#1e40af; }
.tb2 { background:#dcfce7; color:#166534; }
.tb3 { background:#fef9c3; color:#92400e; }
.tb4 { background:#fce7f3; color:#9d174d; }
.tb5 { background:#fee2e2; color:#991b1b; }

/* ── Approval banners ────────────────────────────────────── */
.banner-ok  { background:#f0fdf4; border-left:5px solid #22c55e;
              padding:14px 20px; border-radius:8px;
              font-weight:700; font-size:1.05rem; }
.banner-err { background:#fff1f2; border-left:5px solid #ef4444;
              padding:14px 20px; border-radius:8px;
              font-weight:700; font-size:1.05rem; }
.banner-warn{ background:#fff7ed; border-left:5px solid #f97316;
              padding:14px 20px; border-radius:8px;
              font-weight:700; font-size:1.05rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def _init():
    for k, v in {"result": None, "history": [], "running": False}.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ─────────────────────────────────────────────────────────────────────────────
# NODE CARD HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _card(ph, icon, title, reads, writes, status, note="", attempt=None):
    """
    Render a coloured status card into a Streamlit placeholder.

    status: "idle" | "running" | "ok" | "warn" | "err"
    """
    spin_map = {"idle": "⏳", "running": "🔄", "ok": "✅", "warn": "⚠️", "err": "❌"}
    spin = spin_map.get(status, "⏳")

    attempt_html = ""
    if attempt:
        cls = "fix" if "fix" in attempt.lower() or "improv" in attempt.lower() else ""
        attempt_html = f'<span class="attempt-badge {cls}">{attempt}</span>'

    note_html = f'<div class="nc-note">{note}</div>' if note else ""

    ph.markdown(
        f"""<div class="nc {status}">
          <div class="nc-title">{spin} &nbsp;{icon} &nbsp;{title}{attempt_html}</div>
          <div class="nc-rw">reads  ← {reads}</div>
          <div class="nc-rw">writes → {writes}</div>
          {note_html}
        </div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Code Review Pipeline")
    st.caption("LangGraph · OpenRouter · Streamlit")
    st.divider()

    st.markdown("### 🤖 Model")
    selected_model = st.selectbox(
        "LLM:", list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_KEY),
    )
    st.caption(f"`{AVAILABLE_MODELS[selected_model]}`")
    st.divider()

    st.markdown("### 🔀 Pipeline with Retry")
    st.code(
        "START\n"
        "  ↓\n"
        "🤖 code_generator\n"
        "  ↓\n"
        "🧪 test_generator\n"
        "  ↓\n"
        "▶️  test_runner\n"
        "  ↓  ← conditional edge\n"
        "  ├─ PASS → 👀 code_reviewer\n"
        "  │           ↓  ← conditional edge\n"
        "  │    ├─ APPROVED → END ✅\n"
        "  │    └─ REJECTED → ✨ code_improver\n"
        "  │                      ↓\n"
        "  │              (back to test_generator)\n"
        "  └─ FAIL → 🔧 code_fixer\n"
        "                 ↓\n"
        "         (back to test_generator)\n"
        f"\nMax retries: {MAX_RETRIES}",
        language="text",
    )
    st.divider()

    TEMPLATES = {
        "— choose —": "",
        "Discount Calculator": (
            "Create a function `calculate_discount(price: float, "
            "discount_pct: float) -> float` that returns the discounted price. "
            "discount_pct must be 0–100. Raise ValueError for invalid inputs."
        ),
        "Email Validator": (
            "Create a function `validate_email(email: str) -> bool` "
            "that returns True for a valid email address. "
            "Raise ValueError for non-string input."
        ),
        "Prime Checker": (
            "Create a function `is_prime(n: int) -> bool` "
            "that returns True if n is prime. Raise ValueError for n < 0."
        ),
        "Palindrome Checker": (
            "Create a function `is_palindrome(text: str) -> bool` "
            "that returns True if the string reads the same forwards and backwards, "
            "ignoring case, spaces, and punctuation."
        ),
        "Fibonacci Generator": (
            "Create a generator function `fibonacci(n: int)` "
            "that yields the first n Fibonacci numbers. Raise ValueError if n < 0."
        ),
        "Word Frequency Counter": (
            "Create a function `word_frequency(text: str) -> dict[str, int]` "
            "that returns word counts (lowercase, no punctuation). "
            "Raise ValueError for non-string input."
        ),
    }
    st.markdown("### 🗂️ Templates")
    chosen = st.selectbox("", list(TEMPLATES.keys()), label_visibility="collapsed")
    if chosen != "— choose —":
        st.session_state["_tpl"] = TEMPLATES[chosen]

    st.divider()
    import os
    key_ok = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
    st.success("🔑 API key loaded", icon="✅") if key_ok else st.error("🔑 API key missing", icon="❌")
    st.caption(f"Runs: **{len(st.session_state.history)}**")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER + INPUT
# ─────────────────────────────────────────────────────────────────────────────

st.title("⚙️ Code Review Pipeline")
st.caption(
    "The pipeline generates code, tests it, and reviews it. "
    "If any stage fails, the agent **learns from the error** and retries "
    f"automatically (up to {MAX_RETRIES} times)."
)
st.divider()

default_text = st.session_state.pop("_tpl", "")
feature_request = st.text_area(
    "**Feature Request**",
    value=default_text,
    placeholder=(
        "Describe the Python function you want to build.\n\n"
        "Example: Create a function `calculate_discount(price, discount_pct)` "
        "that returns the discounted price. Raise ValueError for invalid inputs."
    ),
    height=110,
)

col_run, col_clear, _ = st.columns([1, 1, 5])
with col_run:
    run_btn = st.button("▶️ Run Pipeline", type="primary",
                        use_container_width=True, disabled=st.session_state.running)
with col_clear:
    clear_btn = st.button("🗑️ Clear", use_container_width=True,
                          disabled=st.session_state.running)

if clear_btn:
    st.session_state.result = None
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE EXECUTION  — node-by-node with live UI updates and retry loop
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    if not feature_request.strip():
        st.error("Please enter a feature request.")
    else:
        st.session_state.running = True
        st.divider()
        st.markdown("### 🔄 Agent Orchestration")
        st.caption(
            "Each card updates live. 🔧 and ✨ cards appear only if a retry is needed."
        )

        # ── Static node placeholders (always visible) ────────────────────────
        ph1 = st.empty()   # code_generator
        ph2 = st.empty()   # test_generator
        ph3 = st.empty()   # test_runner
        ph4 = st.empty()   # code_reviewer

        # ── Retry node placeholders (only shown when triggered) ──────────────
        ph_fixer    = st.empty()   # code_fixer    (Node 5)
        ph_improver = st.empty()   # code_improver (Node 6)

        status_ph = st.empty()

        # Draw initial idle cards
        _card(ph1, "🤖", "Node 1 — Code Generator",
              "feature_request", "code_content", "idle")
        _card(ph2, "🧪", "Node 2 — Test Generator",
              "code_content", "test_content", "idle")
        _card(ph3, "▶️",  "Node 3 — Test Runner",
              "code_content + test_content", "test_results, tests_passed", "idle")
        _card(ph4, "👀", "Node 4 — Code Reviewer",
              "code_content + test_results", "review_feedback, is_approved", "idle")

        # Initial state (identical to CLI initial_state in app.py)
        state: CodeReviewState = {
            "feature_request": feature_request.strip(),
            "model_key":       selected_model,
            "code_content":    "",
            "test_content":    "",
            "test_results":    "",
            "tests_passed":    False,
            "review_feedback": "",
            "is_approved":     False,
            "retry_count":     0,
            "fix_history":     [],
        }

        model_label = f"<code>{AVAILABLE_MODELS[selected_model]}</code>"

        try:
            # ── NODE 1 : Code Generator ──────────────────────────────────────
            _card(ph1, "🤖", "Node 1 — Code Generator",
                  "feature_request", "code_content", "running",
                  f"Calling {model_label} …")
            state.update(code_generator_node(state))
            lines1 = len(state["code_content"].splitlines())
            _card(ph1, "🤖", "Node 1 — Code Generator",
                  "feature_request", "code_content", "ok",
                  f"Generated <strong>{lines1} lines</strong> of Python code.")

            # ── RETRY LOOP ───────────────────────────────────────────────────
            # We loop: test_generator → test_runner → [fix?] → reviewer → [improve?]
            # until approved OR retry_count >= MAX_RETRIES

            while True:
                attempt_label = (
                    f"attempt {state['retry_count'] + 1}/{MAX_RETRIES}"
                    if state["retry_count"] > 0 else ""
                )

                # ── NODE 2 : Test Generator ──────────────────────────────────
                _card(ph2, "🧪", "Node 2 — Test Generator",
                      "code_content", "test_content", "running",
                      f"Calling {model_label} …",
                      attempt=attempt_label or None)
                state.update(test_generator_node(state))
                fn_count = state["test_content"].count("def test_")
                _card(ph2, "🧪", "Node 2 — Test Generator",
                      "code_content", "test_content", "ok",
                      f"Generated <strong>{fn_count} test functions</strong>.",
                      attempt=attempt_label or None)

                # ── NODE 3 : Test Runner ─────────────────────────────────────
                _card(ph3, "▶️",  "Node 3 — Test Runner",
                      "code_content + test_content",
                      "test_results, tests_passed", "running",
                      "Running pytest in subprocess …",
                      attempt=attempt_label or None)
                state.update(test_runner_node(state))
                p = state["test_results"].count(" PASSED")
                f = state["test_results"].count(" FAILED")
                test_note = (
                    f"<strong>{p} passed</strong>, <strong>{f} failed</strong>"
                    + (" — ✅ All green" if state["tests_passed"] else " — ❌ Some failures")
                )
                _card(ph3, "▶️",  "Node 3 — Test Runner",
                      "code_content + test_content",
                      "test_results, tests_passed",
                      "ok" if state["tests_passed"] else "err",
                      test_note,
                      attempt=attempt_label or None)

                # ── Conditional route after tests ────────────────────────────
                route = route_after_tests(state)

                if route == "__end__":
                    # Exhausted retries with failing tests — stop
                    status_ph.warning(
                        f"⚠️ Tests still failing after {MAX_RETRIES} attempts. "
                        "Showing partial results."
                    )
                    break

                if route == "code_fixer":
                    # Tests failed → fix code and loop back
                    fix_attempt = state["retry_count"] + 1
                    _card(ph_fixer, "🔧", "Node 5 — Code Fixer",
                          "code_content + test_results",
                          "code_content (fixed)", "running",
                          f"Learning from failures and rewriting code … "
                          f"(attempt {fix_attempt}/{MAX_RETRIES})",
                          attempt=f"fix {fix_attempt}")
                    state.update(code_fixer_node(state))
                    fixed_lines = len(state["code_content"].splitlines())
                    _card(ph_fixer, "🔧", "Node 5 — Code Fixer",
                          "code_content + test_results",
                          "code_content (fixed)", "warn",
                          f"Rewrote code → <strong>{fixed_lines} lines</strong>. "
                          f"Retrying tests …",
                          attempt=f"fix {fix_attempt}")
                    # Loop back → test_generator
                    continue

                # route == "code_reviewer" — tests passed, proceed
                # ── NODE 4 : Code Reviewer ───────────────────────────────────
                _card(ph4, "👀", "Node 4 — Code Reviewer",
                      "code_content + test_results",
                      "review_feedback, is_approved", "running",
                      f"Calling {model_label} …")
                state.update(code_reviewer_node(state))
                decision = "APPROVED ✅" if state["is_approved"] else "REJECTED ❌"
                _card(ph4, "👀", "Node 4 — Code Reviewer",
                      "code_content + test_results",
                      "review_feedback, is_approved",
                      "ok" if state["is_approved"] else "err",
                      f"Decision: <strong>{decision}</strong>")

                # ── Conditional route after review ───────────────────────────
                rev_route = route_after_review(state)

                if rev_route == "__end__":
                    if state["is_approved"]:
                        status_ph.success("✅ Pipeline complete! Scroll down for results.")
                    else:
                        status_ph.warning(
                            f"⚠️ Still rejected after {MAX_RETRIES} attempts. "
                            "Showing partial results."
                        )
                    break

                # rev_route == "code_improver" — rejected, improve and loop
                imp_attempt = state["retry_count"] + 1
                _card(ph_improver, "✨", "Node 6 — Code Improver",
                      "code_content + review_feedback",
                      "code_content (improved)", "running",
                      f"Learning from review feedback and rewriting … "
                      f"(attempt {imp_attempt}/{MAX_RETRIES})",
                      attempt=f"improve {imp_attempt}")
                state.update(code_improver_node(state))
                imp_lines = len(state["code_content"].splitlines())
                _card(ph_improver, "✨", "Node 6 — Code Improver",
                      "code_content + review_feedback",
                      "code_content (improved)", "warn",
                      f"Improved code → <strong>{imp_lines} lines</strong>. "
                      f"Re-running tests and review …",
                      attempt=f"improve {imp_attempt}")
                # Loop back → test_generator

            # ── Save result ──────────────────────────────────────────────────
            st.session_state.result = dict(state)
            st.session_state.history.insert(0, {
                "ts":       datetime.now().strftime("%H:%M:%S"),
                "feature":  feature_request.strip()[:60],
                "model":    selected_model,
                "approved": state["is_approved"],
                "retries":  state["retry_count"],
            })

        except EnvironmentError as exc:
            status_ph.error(str(exc))
        except Exception as exc:
            status_ph.error(f"❌ Pipeline error: {exc}")
        finally:
            st.session_state.running = False


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────

result = st.session_state.result

if result:
    st.divider()

    model_str = f"{result['model_key']} · {AVAILABLE_MODELS[result['model_key']]}"
    retry_str = (
        f" &nbsp;·&nbsp; {result['retry_count']} retry attempt(s)"
        if result["retry_count"] > 0 else ""
    )

    if result["is_approved"]:
        st.markdown(
            f'<div class="banner-ok">✅ &nbsp; APPROVED FOR PRODUCTION'
            f'<span style="font-weight:400;font-size:0.8rem;margin-left:12px;color:#166534">'
            f'{model_str}{retry_str}</span></div>',
            unsafe_allow_html=True,
        )
    elif result["tests_passed"]:
        st.markdown(
            f'<div class="banner-warn">⚠️ &nbsp; TESTS PASSED · REVIEW REJECTED'
            f'<span style="font-weight:400;font-size:0.8rem;margin-left:12px;color:#92400e">'
            f'{model_str}{retry_str}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="banner-err">❌ &nbsp; TESTS FAILED · NEEDS REVISION'
            f'<span style="font-weight:400;font-size:0.8rem;margin-left:12px;color:#991b1b">'
            f'{model_str}{retry_str}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tabs = ["🤖 Generated Code", "🧪 Test Cases", "▶️ Test Results", "👀 Code Review"]
    if result.get("fix_history"):
        tabs.append("🔧 Fix History")

    tab_objs = st.tabs(tabs)

    with tab_objs[0]:
        st.markdown('<span class="tb tb1">Node 1/5/6 — Final Code</span>',
                    unsafe_allow_html=True)
        st.caption("Final version of the code after all retries.")
        if result["code_content"]:
            st.code(result["code_content"], language="python")
            c1, c2, c3 = st.columns(3)
            c1.metric("Lines",      len(result["code_content"].splitlines()))
            c2.metric("Functions",  result["code_content"].count("def "))
            c3.metric("Characters", len(result["code_content"]))
        else:
            st.warning("No code available.")

    with tab_objs[1]:
        st.markdown('<span class="tb tb2">Node 2 — Test Cases</span>',
                    unsafe_allow_html=True)
        st.caption("Final test suite (last generated before approval).")
        if result["test_content"]:
            st.code(result["test_content"], language="python")
            c1, c2 = st.columns(2)
            c1.metric("Test functions", result["test_content"].count("def test_"))
            c2.metric("Lines",          len(result["test_content"].splitlines()))
        else:
            st.warning("No tests available.")

    with tab_objs[2]:
        st.markdown('<span class="tb tb3">Node 3 — Test Results</span>',
                    unsafe_allow_html=True)
        st.caption("Output from the last pytest run.")
        if result["tests_passed"]:
            st.success("✅ All tests PASSED")
        else:
            st.error("❌ Some tests FAILED")
        if result["test_results"]:
            st.code(result["test_results"], language="text")

    with tab_objs[3]:
        st.markdown('<span class="tb tb4">Node 4 — Code Review</span>',
                    unsafe_allow_html=True)
        st.caption("Final review from the LLM.")
        if result["review_feedback"]:
            st.markdown(result["review_feedback"])
        else:
            st.warning("No review yet (tests may still be failing).")

    if result.get("fix_history") and len(tab_objs) > 4:
        with tab_objs[4]:
            st.markdown('<span class="tb tb5">Fix / Improve History</span>',
                        unsafe_allow_html=True)
            st.caption(
                "Every time the agent learned from a failure and rewrote the code."
            )
            for entry in result["fix_history"]:
                reason_label = (
                    "🔧 Test Failure Fix"
                    if entry["reason"] == "test_failure"
                    else "✨ Review Rejection Fix"
                )
                with st.expander(
                    f"Attempt {entry['attempt']} — {reason_label}"
                ):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Before (first 300 chars)**")
                        st.code(entry["old_code"], language="python")
                    with col_b:
                        st.markdown("**After (first 300 chars)**")
                        st.code(entry["new_code"], language="python")

    # Full state dump
    st.divider()
    with st.expander("📦 Complete final_state (graph.invoke result)"):
        display = {
            k: (v[:500] + " …[truncated]" if isinstance(v, str) and len(v) > 500 else v)
            for k, v in result.items()
        }
        st.json(display)


# ─────────────────────────────────────────────────────────────────────────────
# RUN HISTORY
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.history:
    st.divider()
    st.markdown("### 📜 Run History")
    for run in st.session_state.history:
        icon = "✅" if run["approved"] else "❌"
        retries = f" · {run['retries']} retry" if run["retries"] else ""
        st.caption(
            f"{icon} `{run['ts']}` · **{run['model']}**{retries} · {run['feature']}"
        )