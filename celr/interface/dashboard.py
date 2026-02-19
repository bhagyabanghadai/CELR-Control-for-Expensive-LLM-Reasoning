"""
Cerebro Dashboard — Phase 9: The Singularity Update
====================================================
UI Designer + Frontend Dev: Real-time "War Room" interface.

Run with:
    streamlit run celr/interface/dashboard.py
    # or via CLI:
    celr --ui

Panels:
  1. 🧠 Hive-Mind Council  – live debate stream
  2. 💰 Budget Burn Chart  – $ spent per step as a live chart
  3. 🔬 Neural Activations – Nano-Cortex state vector bar chart
  4. 📋 Agent Log          – execution history feed
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import streamlit as st

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CELR · Cerebro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (glassmorphism + dark) ──────────────────────────────────────
st.markdown("""
<style>
/* Dark base */
[data-testid="stAppViewContainer"] { background: #0a0a0f; color: #e2e8f0; }
[data-testid="stSidebar"]           { background: #0d0d1a; }

/* Card style */
.card {
  background: rgba(15,15,30,0.85);
  border: 1px solid rgba(99,102,241,0.25);
  border-radius: 12px;
  padding: 1.2rem 1.5rem;
  margin-bottom: 1rem;
  backdrop-filter: blur(10px);
}

/* Verdict badges */
.badge-approve  { background:#052e16; color:#4ade80; padding:2px 10px; border-radius:20px; font-size:0.8rem; }
.badge-reject   { background:#450a0a; color:#f87171; padding:2px 10px; border-radius:20px; font-size:0.8rem; }
.badge-escalate { background:#1c1917; color:#fbbf24; padding:2px 10px; border-radius:20px; font-size:0.8rem; }

/* Metrics */
[data-testid="metric-container"] { 
  background:rgba(15,15,30,0.7);
  border:1px solid rgba(99,102,241,0.2);
  border-radius:8px;
  padding:0.5rem 1rem;
}

h1, h2, h3 { color: #a5b4fc !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────

LOG_DIR = Path(".celr_logs")


def _read_latest_log() -> Optional[dict]:
    """Read the most recent CELR trajectory log."""
    traj_dir = LOG_DIR / "Traj"
    if not traj_dir.exists():
        return None
    files = sorted(traj_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return None


def _badge(verdict: str) -> str:
    cls = f"badge-{verdict.lower()}"
    return f'<span class="{cls}">{verdict}</span>'


# ── Layout ─────────────────────────────────────────────────────────────────

# Sidebar
with st.sidebar:
    st.markdown("## 🧠 Cerebro")
    st.caption("CELR · Phase 9 · Real-Time War Room")
    st.divider()
    refresh_rate = st.slider("Refresh (s)", 1, 30, 3)
    show_activations = st.toggle("Show Neural Activations", value=True)
    show_council = st.toggle("Show Council Debates", value=True)
    st.divider()
    st.markdown("**Council Members**")
    st.markdown("- 🔴 Skeptic (`llama3.2`)")
    st.markdown("- 🟢 Optimist (`llama3.2`)")
    st.markdown("- 🔵 Realist (`llama3.2`)")
    st.markdown("- ⭐ Chairman (`llama3.2`)")

# ── Header ─────────────────────────────────────────
st.markdown("# 🧠 CELR · Cerebro — Adaptive Intelligence War Room")
st.caption("Real-time monitoring of the Hive-Mind Council and Nano-Cortex Decision Engine")

# ── Top Metrics ────────────────────────────────────
log = _read_latest_log()

col1, col2, col3, col4 = st.columns(4)
if log:
    steps = log.get("steps", [])
    status = log.get("status", "RUNNING")
    col1.metric("Status",        status,      delta=None)
    col2.metric("Steps Done",    len(steps),  delta=None)
    col3.metric("Budget Used",  f"${log.get('cost_usd', 0):.4f}", delta=None)
    col4.metric("Retries",       log.get("total_retries", 0), delta=None)
else:
    col1.metric("Status",    "Waiting",  "No log yet")
    col2.metric("Steps",     "–",        "")
    col3.metric("Budget",    "$0.0000",  "")
    col4.metric("Retries",   "–",        "")

st.divider()

# ── Main panels ────────────────────────────────────

left, right = st.columns([3, 2])

# ── LEFT: Council Debate Stream ────────────────────
with left:
    if show_council:
        st.markdown("### ⚖️ Hive-Mind Council — Live Debate")
        council_log = []
        if log:
            council_log = log.get("council_debates", [])

        if council_log:
            for debate in reversed(council_log[-8:]):   # last 8 debates
                with st.container():
                    st.markdown(f"""
<div class="card">
  <b>Proposal:</b> {debate.get('proposal','')[:120]}…<br/>
  {''.join(f"<b>{v['member_id']}</b>: {_badge(v['verdict'])} {v['reasoning'][:80]}<br/>" for v in debate.get('votes',[]))}
  <br/><b>Final: </b>{_badge(debate.get('final_verdict','?'))} {debate.get('final_reasoning','')[:100]}
</div>""", unsafe_allow_html=True)
        else:
            st.info("No council debates yet. Run a task to see the Hive-Mind think.")

    # Agent Log
    st.markdown("### 📋 Agent Execution Log")
    if log:
        history = log.get("execution_history", [])
        for entry in reversed(history[-20:]):
            icon = "✅" if "complete" in entry.lower() else ("❌" if "fail" in entry.lower() else "📌")
            st.markdown(f"`{icon}` {entry}")
    else:
        st.info("Start a CELR task to see the live log here.")

# ── RIGHT: Charts ──────────────────────────────────
with right:
    # Budget Burn Chart
    st.markdown("### 💰 Budget Burn")
    if log:
        steps_data = log.get("steps", [])
        costs = [s.get("cost_usd", 0.0) for s in steps_data]
        cumulative = [sum(costs[:i+1]) for i in range(len(costs))]
        if cumulative:
            import pandas as pd
            df = pd.DataFrame({
                "Step": list(range(1, len(cumulative)+1)),
                "Cumulative Cost ($)": cumulative,
            }).set_index("Step")
            st.line_chart(df, color="#6366f1")
        else:
            st.caption("No cost data yet.")
    else:
        st.caption("No data yet.")

    # Neural Activations
    if show_activations:
        st.markdown("### 🔬 Nano-Cortex State Vector")
        state_labels = [
            "Budget Left", "Retry Rate", "Difficulty", "Tool Use",
            "Success Rate", "Escalation", "Error Rate", "Time Pressure"
        ]
        if log:
            steps_data = log.get("steps", [])
            last_state = None
            for s in reversed(steps_data):
                if "state_vector" in s:
                    last_state = s["state_vector"]
                    break
            if last_state:
                import pandas as pd
                df = pd.DataFrame({
                    "Dimension": state_labels[:len(last_state)],
                    "Activation": last_state,
                }).set_index("Dimension")
                st.bar_chart(df, color="#a78bfa")
            else:
                st.caption("No state data available.")
        else:
            st.caption("Waiting for first CELR run…")

# ── Auto-refresh ───────────────────────────────────
time.sleep(0.1)   # avoid busy-loop on first render
st.markdown(f"<p style='color:#475569;font-size:.75rem;'>Auto-refreshes every {refresh_rate}s · {time.strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)

# Streamlit native auto-rerun
st.rerun() if st.session_state.get("_auto_refresh") else None

# Bootstrap auto-refresh key on first load
if "_auto_refresh" not in st.session_state:
    st.session_state["_auto_refresh"] = True
    time.sleep(refresh_rate)
    st.rerun()
