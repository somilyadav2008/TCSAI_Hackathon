# maintainance.py
# Fast banking maintenance workflow wrapped as a single-input JSON Tool for a Conversational agent.

import re
import json
from typing import Dict, List

# ---- LangChain (updated imports) ----
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferWindowMemory

def read_text_file(file_path: str) -> str:
    """
    Reads the content of a text file and returns it as a string.

    :param file_path: Path to the .txt file
    :return: Content of the file as a string
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"Error while reading file: {e}"

# ===================== Sample Data (fallback) ===================== 
data = r"C:\Users\Administrator\Documents\Testing Project\logs.txt"
data2 = r"C:\Users\Administrator\Documents\Testing Project\manual.txt"

SAMPLE_LOGS = read_text_file(data).strip() 
print("------------------")
print(SAMPLE_LOGS) 
print("------------------")
# SAMPLE_LOGS.strip()
SAMPLE_MANUAL = read_text_file(data2).strip()
# SAMPLE_MANUAL.strip()
print(SAMPLE_MANUAL)
DEFAULT_PROBLEM = "Users report frequent transaction failures during peak hours with intermittent errors and slow response."
DEFAULT_SCENARIO = (
    "A transaction submitted through the banking portal intermittently fails "
    "with internal error and timeout messages in the logs. The system must diagnose, "
    "resolve, and validate the resolution."
).strip()

# ===================== Fast Workflow (no agents inside) =====================
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
        import numpy as np
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

# ===================== Tool (single-input JSON string) =====================
def maintenance_tool_entry(payload_json: str) -> str:
    """
    INPUT: JSON string with keys:
      problem:str, logs:str, manual:str, scenario:str, top_n:int (opt)
    OUTPUT: JSON string with keys: log_analysis, diagnosis, resolution, validation, report
    """
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

# ===================== Agent wiring =====================
def make_agent():
    llm = Ollama(model="llama-3.2-3b-it", temperature=0)  # pass params directly
    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools=[MaintenanceWorkflowTool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  # requires single-input tool → we used JSON tool
        memory=memory,
        max_iterations=2,
        verbose=False,
    )
    return agent

# ===================== CLI-style demo =====================
if __name__ == "__main__":
    agent = make_agent()

    payload = json.dumps({
        "problem": DEFAULT_PROBLEM,
        "logs": SAMPLE_LOGS,
        "manual": SAMPLE_MANUAL,
        "scenario": DEFAULT_SCENARIO,
        "top_n": 3
    }, ensure_ascii=False)

    prompt = (
        "Call the MaintenanceWorkflow tool. Use the following JSON EXACTLY as the tool input:\n"
        f"{payload}\n"
        "Return only the tool's JSON output."
    )

    # Conversational agent expects a single 'input' key
    result = agent.invoke({"input": prompt})
    print(result)
