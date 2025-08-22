# app.py
# Streamlit UI for your maintenance workflow (LangChain agent + single-input JSON Tool).
# Upload logs/manual files, tweak params, run, and download results.

import json
import re
from typing import Dict, List

import streamlit as st

# LangChain (non-deprecated imports)
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferWindowMemory


# ===================== Core workflow (same as your maintainance.py) =====================

def log_analysis_fn(logs: str) -> str:
    issues = []
    t = logs.lower()
    if "timeout" in t:
        issues.append("Network timeout or connectivity failure")
    if "error 500" in t:
        issues.append("Internal server error")
    if "failed" in t or "transaction failed" in t:
        issues.append("Transaction failure")
    if "deadlock" in t:
        issues.append("Database deadlock leading to rollback")
    if not issues:
        issues.append("Unclassified issue")
    return "; ".join(sorted(set(issues)))

def diagnostic_fn(input_text: str) -> str:
    return (
        "Diagnosis:\n"
        f"{input_text}\n"
        "→ Likely contention + connectivity issues during peak load (DB deadlocks, external API timeouts), "
        "exacerbated by missing idempotency/unique txn guarantees."
    )

def resolution_fn(input_text: str) -> str:
    return (
        "Resolution Plan:\n"
        "1) Capacity & Contention:\n"
        "   - Scale DB read replicas; review locks/order-of-operations in stored procedures.\n"
        "   - Add/review indexes; move heavy reporting jobs off-peak.\n"
        "2) Idempotency & Duplicates:\n"
        "   - Enforce unique transaction_id at middleware; DB UNIQUE(transaction_id).\n"
        "   - Implement retry with backoff and idempotency keys end-to-end.\n"
        "3) External APIs:\n"
        "   - Monitor latency; verify firewall/DNS; auto-failover to backup gateway.\n"
        "4) Monitoring & Alerts:\n"
        "   - Add slow-query log dashboards; alert on deadlocks/timeouts.\n"
        "5) Runbook:\n"
        "   - Apply steps from retrieved manual sections; validate after each change."
    )

def validation_fn(input_text: str) -> str:
    return (
        "Validation:\n"
        "• Re-run peak-hour synthetic load (same traffic profile).\n"
        "• Verify: deadlocks ↓, timeout errors ↓, duplicate transactions = 0.\n"
        "• Check gateway failover works; confirm E2E success rate ≥ 99.9%.\n"
        "• Confirm reconciliation/interest calculations remain correct."
    )

EMOJI_HEAD_RE = re.compile(r"(?=^([🔐💳🏦📂🗄💰📊📢🔒].+)$)", re.MULTILINE)
SECTION_FALLBACK_RE = re.compile(
    r"(?=^([^\n].*?Login Service|Transaction Engine|Payment Gateway|Loan Processing|Database Layer|Core Banking|Reporting Module|Notification Service|Security & Audit).*$)",
    re.MULTILINE,
)

def chunk_manual(manual: str) -> Dict[str, str]:
    if EMOJI_HEAD_RE.search(manual):
        parts = EMOJI_HEAD_RE.split(manual)
    else:
        parts = SECTION_FALLBACK_RE.split(manual)
    chunks = {}
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        chunks[header] = f"{header}\n{body}".strip()
    if not chunks:
        chunks["Manual"] = manual
    return chunks

def simple_keyword_retrieve(chunks: Dict[str, str], query: str, top_n: int = 3) -> List[str]:
    keywords = re.findall(r"[A-Za-z0-9_]+", query.lower())
    scored = []
    for h, txt in chunks.items():
        t = txt.lower()
        score = sum(t.count(k) for k in keywords if len(k) > 3)
        scored.append((score, h, txt))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = [txt for (score, h, txt) in scored[:top_n] if score > 0]
    return top or list(chunks.values())[:min(top_n, len(chunks))]

def tfidf_retrieve(chunks: Dict[str, str], query: str, top_n: int = 3) -> List[str]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np  # noqa: F401
    except Exception:
        return simple_keyword_retrieve(chunks, query, top_n)
    texts = list(chunks.values())
    vect = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X = vect.fit_transform(texts + [query])
    qv = X[-1]
    sims = (X[:-1] @ qv.T).toarray().ravel()
    idx = sims.argsort()[::-1][:top_n]
    picked = [texts[i] for i in idx if sims[i] > 0]
    return picked or simple_keyword_retrieve(chunks, query, top_n)

def full_maintenance_workflow_fast(
    problem: str,
    logs: str,
    manual: str,
    scenario: str,
    top_n: int = 3,
) -> Dict[str, str]:
    chunks = chunk_manual(manual)
    log_issues = log_analysis_fn(logs)
    diag_context = "\n\n".join(tfidf_retrieve(chunks, f"{problem} {log_issues}", top_n=top_n))
    diagnosis_input = f"Problem: {problem}\nLogAnalysis: {log_issues}\nRelevant Manual Sections:\n{diag_context}"
    diagnosis = diagnostic_fn(diagnosis_input)
    res_context = "\n\n".join(tfidf_retrieve(chunks, diagnosis, top_n=top_n))
    resolution_input = f"{diagnosis}\n\nRelevant Manual Sections:\n{res_context}"
    resolution = resolution_fn(resolution_input)
    validation_input = f"Resolution: {resolution}\nScenario: {scenario}"
    validation = validation_fn(validation_input)
    compact_report = (
        "=== Maintenance Summary ===\n"
        f"Problem:\n{problem}\n\n"
        f"Log Analysis:\n{log_issues}\n\n"
        f"Diagnosis:\n{diagnosis}\n\n"
        f"Resolution:\n{resolution}\n\n"
        f"Validation Plan:\n{validation}\n"
    )
    return {
        "log_analysis": log_issues,
        "diagnosis": diagnosis,
        "resolution": resolution,
        "validation": validation,
        "report": compact_report,
    }

# Single-input JSON tool wrapper (required by Conversational agent)
def maintenance_tool_entry(payload_json: str) -> str:
    data = json.loads(payload_json)
    results = full_maintenance_workflow_fast(
        problem=data.get("problem", ""),
        logs=data.get("logs", ""),
        manual=data.get("manual", ""),
        scenario=data.get("scenario", ""),
        top_n=int(data.get("top_n", 3)),
    )
    return json.dumps(results, ensure_ascii=False, indent=2)

MaintenanceWorkflowTool = Tool(
    name="MaintenanceWorkflow",
    description=(
        "Run the fast maintenance workflow. INPUT MUST BE A JSON STRING with keys: "
        "problem, logs, manual, scenario, top_n. "
        "Returns JSON string with log_analysis, diagnosis, resolution, validation, report."
    ),
    func=maintenance_tool_entry,
)


# ===================== Streamlit UI =====================

st.set_page_config(page_title="Maintenance Workflow Agent", page_icon="🛠️", layout="wide")

st.title("🛠️ Maintenance Workflow Agent")
st.write("Upload **logs** and **manual** files, set parameters, and run the workflow via a LangChain agent.")

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Ollama model", value="llama-3.2-3b-it", help="Ensure the model is pulled in Ollama.")
    temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    top_n = st.slider("Top-N manual sections", 1, 10, 3)
    st.caption("Tip: Keep Top-N small for speed.")

st.subheader("1) Upload files")
col1, col2 = st.columns(2)
with col1:
    logs_file = st.file_uploader("Logs file (.txt, .log)", type=["txt", "log"])
with col2:
    manual_file = st.file_uploader("Manual file (.txt, .md)", type=["txt", "md"])

st.subheader("2) Describe the problem & scenario")
default_problem = "Users report frequent transaction failures during peak hours with intermittent errors and slow response."
default_scenario = ("A transaction submitted through the banking portal intermittently fails "
                    "with internal error and timeout messages in the logs. The system must diagnose, "
                    "resolve, and validate the resolution.")
problem = st.text_area("Problem", value=default_problem, height=80)
scenario = st.text_area("Scenario", value=default_scenario, height=80)

run_btn = st.button("🚀 Run Maintenance Workflow")

# ===================== Error helpers =====================

MAX_SIZE = 5 * 1024 * 1024  # 5 MB

def read_upload(file) -> str:
    """Decode uploaded file to UTF-8 text with graceful fallback."""
    if not file:
        return ""
    if file.size > MAX_SIZE:
        raise ValueError(f"File too large ({file.size} bytes). Limit is {MAX_SIZE} bytes.")
    try:
        data = file.read()
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            # fallback: latin-1 to avoid crash; not ideal but ensures app stays responsive
            return data.decode("latin-1")
    finally:
        file.seek(0)  # reset pointer for any later use

@st.cache_resource(show_spinner=False)
def make_agent_cached(model_name: str, temperature: float):
    """Build the agent once per model/temperature pair."""
    llm = Ollama(model=model_name, temperature=temperature)
    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools=[MaintenanceWorkflowTool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  # single-input tool
        memory=memory,
        max_iterations=2,
        verbose=False,
    )
    return agent

# ===================== Run logic =====================

if run_btn:
    try:
        # Validate inputs
        if logs_file is None or manual_file is None:
            st.error("Please upload **both** a logs file and a manual file.")
            st.stop()

        logs_text = read_upload(logs_file)
        manual_text = read_upload(manual_file)

        if not logs_text.strip():
            st.error("The logs file appears to be empty or unreadable.")
            st.stop()
        if not manual_text.strip():
            st.error("The manual file appears to be empty or unreadable.")
            st.stop()
        if not problem.strip():
            st.error("Please provide a problem description.")
            st.stop()
        if not scenario.strip():
            st.error("Please provide a scenario.")
            st.stop()

        # Build payload JSON for the tool
        payload = json.dumps({
            "problem": problem,
            "logs": logs_text,
            "manual": manual_text,
            "scenario": scenario,
            "top_n": int(top_n),
        }, ensure_ascii=False)

        # Build/run the agent
        with st.spinner("Running agent…"):
            agent = make_agent_cached(model_name, float(temperature))
            prompt = (
                "Call the MaintenanceWorkflow tool. Use the following JSON EXACTLY as the tool input:\n"
                f"{payload}\n"
                "Return only the tool's JSON output."
            )
            result = agent.invoke({"input": prompt})

        # Agent sometimes returns dict or string; handle both
        tool_json = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)
        try:
            parsed = json.loads(tool_json)
        except Exception:
            # Some LLMs wrap again; try to extract JSON
            try:
                start = tool_json.find("{")
                end = tool_json.rfind("}") + 1
                parsed = json.loads(tool_json[start:end])
            except Exception:
                st.error("Could not parse the agent output as JSON. Showing raw output below.")
                st.code(tool_json, language="json")
                st.stop()

        st.success("Workflow completed.")
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Diagnosis")
            st.write(parsed.get("diagnosis", ""))
            st.subheader("Resolution")
            st.write(parsed.get("resolution", ""))
        with colB:
            st.subheader("Log Analysis")
            st.write(parsed.get("log_analysis", ""))
            st.subheader("Validation Plan")
            st.write(parsed.get("validation", ""))

        st.subheader("Report")
        st.code(parsed.get("report", ""), language="markdown")

        # Downloads
        st.download_button(
            label="⬇️ Download JSON",
            data=json.dumps(parsed, ensure_ascii=False, indent=2),
            file_name="maintenance_results.json",
            mime="application/json",
        )
        st.download_button(
            label="⬇️ Download Report (.txt)",
            data=parsed.get("report", ""),
            file_name="maintenance_report.txt",
            mime="text/plain",
        )

    except ValueError as ve:
        st.error(f"Input error: {ve}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.exception(e)


# ===================== Footer =====================
st.caption(
    "Powered by LangChain (Conversational ReAct agent + single-input JSON tool) "
    "and a fast local TF-IDF retrieval over your manual."
)
