import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import re
from typing import List, Dict, Any
import hashlib
from dataclasses import dataclass
from enum import Enum
import time

# Configure page

st.set_page_config(
page_title=“🏦 Banking Maintenance AI Resolver”,
page_icon=“🏦”,
layout=“wide”,
initial_sidebar_state=“expanded”
)

# Custom CSS for beautiful UI

st.markdown(”””

<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
    }
    
    .issue-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .resolution-card {
        background: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .chat-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        max-height: 400px;
        overflow-y: auto;
    }
    
    .user-message {
        background: #e3f2fd;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: right;
    }
    
    .bot-message {
        background: #f5f5f5;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>

“””, unsafe_allow_html=True)

class IssueStatus(Enum):
UNRESOLVED = “Unresolved”
RESOLVED = “Resolved”
ESCALATED = “Escalated”
MONITORING = “Monitoring”
AUTO_RECOVERED = “Auto-recovered”
AUTO_RETRY = “Auto-retry”

class IssueSeverity(Enum):
LOW = “Low”
MEDIUM = “Medium”
HIGH = “High”
CRITICAL = “Critical”

@dataclass
class MaintenanceIssue:
timestamp: datetime
module: str
description: str
error_code: str
status: IssueStatus
severity: IssueSeverity = IssueSeverity.MEDIUM
resolution_steps: List[str] = None
resolved_by: str = None
resolution_time: datetime = None

# Initialize session state for agent memory

if ‘conversation_memory’ not in st.session_state:
st.session_state.conversation_memory = []
if ‘active_issues’ not in st.session_state:
st.session_state.active_issues = []
if ‘resolved_issues’ not in st.session_state:
st.session_state.resolved_issues = []
if ‘agent_knowledge_base’ not in st.session_state:
st.session_state.agent_knowledge_base = {}

class BankingMaintenanceAgent:
def **init**(self):
self.knowledge_base = self._initialize_knowledge_base()
self.memory = st.session_state.conversation_memory

```
def _initialize_knowledge_base(self):
    """Initialize the RAG knowledge base with troubleshooting manuals"""
    return {
        "Login Service": {
            "common_issues": ["Session timeout", "Account lockout", "Authentication failure"],
            "error_codes": {
                "ERR1001": "User session timeout too early",
                "ERR1002": "Multiple failed login attempts"
            },
            "resolutions": {
                "ERR1001": [
                    "Check session policy in auth_config.yaml",
                    "Verify session DB (Redis/Memcache) is not evicting entries",
                    "Restart session management service",
                    "Update session timeout configuration"
                ],
                "ERR1002": [
                    "Verify lockout threshold in user management",
                    "Manually unlock account in user_mgmt table",
                    "Check for false positive detections",
                    "Review security policies"
                ]
            }
        },
        "Transaction Engine": {
            "common_issues": ["Deadlocks", "Duplicate transactions", "Sync delays"],
            "error_codes": {
                "ERR2001": "Transaction rollback due to deadlock",
                "ERR2002": "Duplicate transaction detected",
                "ERR2003": "Cross-branch transaction sync delay"
            },
            "resolutions": {
                "ERR2001": [
                    "Identify blocking queries using SHOW ENGINE INNODB STATUS",
                    "Optimize stored procedures for order of operations",
                    "Implement retry mechanism with exponential backoff",
                    "Review transaction isolation levels"
                ],
                "ERR2002": [
                    "Enforce unique transaction ID at middleware",
                    "Add DB unique constraint on transaction_id",
                    "Run reconciliation script for duplicate cleanup",
                    "Update client-side duplicate prevention"
                ]
            }
        },
        "Payment Gateway": {
            "common_issues": ["API timeouts", "Checksum errors", "Network connectivity"],
            "error_codes": {
                "ERR3001": "Timeout while connecting to VISA API",
                "ERR3002": "Invalid checksum in payment packet"
            },
            "resolutions": {
                "ERR3001": [
                    "Run ping/traceroute to payment endpoint",
                    "Verify firewall outbound port 443 rules",
                    "Switch to backup gateway if downtime > 30s",
                    "Check DNS resolution"
                ],
                "ERR3002": [
                    "Validate key rotation logs",
                    "Compare checksum algorithm versions",
                    "Re-sync secret keys with payment provider",
                    "Verify payload integrity"
                ]
            }
        },
        "Database Layer": {
            "common_issues": ["High CPU usage", "Backup failures", "Deadlocks"],
            "error_codes": {
                "ERR5001": "High CPU usage detected (>90%)",
                "ERR5002": "Deadlock in customer_accounts table",
                "ERR5003": "Backup job failed – insufficient storage"
            },
            "resolutions": {
                "ERR5001": [
                    "Run slow query log analysis",
                    "Create missing indexes",
                    "Shift reporting jobs to off-peak hours",
                    "Optimize query performance"
                ],
                "ERR5003": [
                    "Clear old logs and backups",
                    "Add storage or move backups to cloud",
                    "Automate retention policy",
                    "Compress backup files"
                ]
            }
        }
    }

def diagnose_issue(self, issue_description: str, error_code: str = None) -> Dict[str, Any]:
    """Main diagnostic function using RAG approach"""
    diagnosis = {
        "identified_module": None,
        "severity": IssueSeverity.MEDIUM,
        "root_cause": "Unknown",
        "resolution_steps": [],
        "estimated_time": "Unknown",
        "escalation_needed": False
    }
    
    # Extract module from issue description
    for module in self.knowledge_base.keys():
        if module.lower() in issue_description.lower():
            diagnosis["identified_module"] = module
            break
    
    # If error code provided, use it for precise diagnosis
    if error_code:
        for module, data in self.knowledge_base.items():
            if error_code in data.get("error_codes", {}):
                diagnosis["identified_module"] = module
                diagnosis["root_cause"] = data["error_codes"][error_code]
                diagnosis["resolution_steps"] = data["resolutions"].get(error_code, [])
                break
    
    # Determine severity based on keywords
    critical_keywords = ["deadlock", "backup failed", "high cpu", "timeout"]
    high_keywords = ["duplicate", "sync delay", "checksum"]
    
    if any(keyword in issue_description.lower() for keyword in critical_keywords):
        diagnosis["severity"] = IssueSeverity.CRITICAL
        diagnosis["estimated_time"] = "30-60 minutes"
    elif any(keyword in issue_description.lower() for keyword in high_keywords):
        diagnosis["severity"] = IssueSeverity.HIGH
        diagnosis["estimated_time"] = "15-30 minutes"
    else:
        diagnosis["severity"] = IssueSeverity.MEDIUM
        diagnosis["estimated_time"] = "10-20 minutes"
    
    # Check if escalation is needed
    escalation_keywords = ["security", "fraud", "data loss", "compliance"]
    if any(keyword in issue_description.lower() for keyword in escalation_keywords):
        diagnosis["escalation_needed"] = True
    
    return diagnosis

def chat_response(self, user_message: str) -> str:
    """Generate chat response with memory"""
    # Add to conversation memory
    self.memory.append({"role": "user", "content": user_message, "timestamp": datetime.now()})
    
    # Simple keyword-based responses (in real implementation, use LLM)
    if "help" in user_message.lower():
        response = "I can help you diagnose and resolve banking application issues. You can:\n1. Report a new issue\n2. Check existing issue status\n3. Get step-by-step resolution guides\n4. Escalate critical issues"
    elif "issue" in user_message.lower() or "problem" in user_message.lower():
        response = "I understand you're reporting an issue. Please provide the error code and description, and I'll diagnose it for you."
    elif "thank" in user_message.lower():
        response = "You're welcome! I'm here to help resolve your banking application maintenance issues quickly and efficiently."
    else:
        # Analyze the message for potential issues
        diagnosis = self.diagnose_issue(user_message)
        if diagnosis["identified_module"]:
            response = f"I've analyzed your message and identified this as a {diagnosis['identified_module']} issue with {diagnosis['severity'].value} severity. Would you like me to provide the resolution steps?"
        else:
            response = "I'm analyzing your request. Could you provide more details about the specific module or error code involved?"
    
    # Add response to memory
    self.memory.append({"role": "assistant", "content": response, "timestamp": datetime.now()})
    return response
```

# Initialize agent

agent = BankingMaintenanceAgent()

# Sample data for demonstration

def load_sample_data():
“”“Load sample maintenance logs”””
sample_logs = [
{“timestamp”: “2025-08-22 09:45:12”, “module”: “Login Service”, “issue”: “User session timeout too early”, “error_code”: “ERR1001”, “status”: “Unresolved”},
{“timestamp”: “2025-08-22 10:07:18”, “module”: “Login Service”, “issue”: “Multiple failed login attempts”, “error_code”: “ERR1002”, “status”: “Resolved”},
{“timestamp”: “2025-08-22 10:15:07”, “module”: “Transaction Engine”, “issue”: “Transaction rollback due to deadlock”, “error_code”: “ERR2001”, “status”: “Auto-recovered”},
{“timestamp”: “2025-08-22 10:22:42”, “module”: “Transaction Engine”, “issue”: “Duplicate transaction detected”, “error_code”: “ERR2002”, “status”: “Escalated”},
{“timestamp”: “2025-08-22 10:36:10”, “module”: “Payment Gateway”, “issue”: “Timeout while connecting to VISA API”, “error_code”: “ERR3001”, “status”: “Escalated”},
{“timestamp”: “2025-08-22 11:03:07”, “module”: “Database Layer”, “issue”: “High CPU usage detected (>90%)”, “error_code”: “ERR5001”, “status”: “Monitoring”},
{“timestamp”: “2025-08-22 11:17:29”, “module”: “Database Layer”, “issue”: “Backup job failed – insufficient storage”, “error_code”: “ERR5003”, “status”: “Escalated”},
]

```
df = pd.DataFrame(sample_logs)
df['timestamp'] = pd.to_datetime(df['timestamp'])
return df
```

# Main Application Layout

def main():
# Header
st.markdown(”””
<div class="main-header">
<h1>🏦 Banking Application Maintenance AI Resolver</h1>
<p>Intelligent Issue Diagnosis & Resolution with Memory & RAG</p>
</div>
“””, unsafe_allow_html=True)

```
# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.title("🔧 Control Panel")
    
    page = st.radio("Navigate to:", [
        "📊 Dashboard",
        "🤖 AI Chatbot",
        "🔍 Issue Diagnosis",
        "📝 Report New Issue",
        "📈 Analytics"
    ])
    
    st.markdown("---")
    st.subheader("🎯 Quick Stats")
    
    # Load sample data
    df = load_sample_data()
    
    total_issues = len(df)
    unresolved = len(df[df['status'] == 'Unresolved'])
    escalated = len(df[df['status'] == 'Escalated'])
    
    st.metric("Total Issues Today", total_issues)
    st.metric("Unresolved", unresolved, delta=f"-{total_issues-unresolved}")
    st.metric("Escalated", escalated, delta="⚠️" if escalated > 0 else "✅")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main content based on page selection
if page == "📊 Dashboard":
    show_dashboard(df)
elif page == "🤖 AI Chatbot":
    show_chatbot()
elif page == "🔍 Issue Diagnosis":
    show_diagnosis()
elif page == "📝 Report New Issue":
    show_report_issue()
elif page == "📈 Analytics":
    show_analytics(df)
```

def show_dashboard(df):
“”“Display main dashboard”””
st.header(“📊 Real-Time Maintenance Dashboard”)

```
# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>🔴 Critical Issues</h3>
        <h2>3</h2>
        <p>Requires immediate attention</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>🟡 Pending Issues</h3>
        <h2>7</h2>
        <p>In progress</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>🟢 Resolved Today</h3>
        <h2>12</h2>
        <p>Successfully fixed</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3>⚡ Avg Resolution</h3>
        <h2>23min</h2>
        <p>Average fix time</p>
    </div>
    """, unsafe_allow_html=True)

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Issues by Module")
    module_counts = df['module'].value_counts()
    fig = px.pie(values=module_counts.values, names=module_counts.index, 
                color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Status Distribution")
    status_counts = df['status'].value_counts()
    fig = px.bar(x=status_counts.index, y=status_counts.values,
                color=status_counts.values, color_continuous_scale='RdYlGn_r')
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Recent issues table
st.subheader("🕐 Recent Issues")
st.dataframe(df.sort_values('timestamp', ascending=False), use_container_width=True)
```

def show_chatbot():
“”“Display AI chatbot interface”””
st.header(“🤖 AI Maintenance Assistant”)

```
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
            padding: 1rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
    <h3>💬 Chat with your intelligent maintenance assistant</h3>
    <p>I can help diagnose issues, provide resolution steps, and learn from our conversations!</p>
</div>
""", unsafe_allow_html=True)

# Chat container
chat_container = st.container()

with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display conversation history
    for msg in st.session_state.conversation_memory[-10:]:  # Show last 10 messages
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input("Type your message...", placeholder="e.g., Login service is timing out users")
    with col2:
        send_button = st.form_submit_button("Send 💬")
    
    if send_button and user_input:
        response = agent.chat_response(user_input)
        st.rerun()

# Quick action buttons
st.subheader("🚀 Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔍 Diagnose Issue"):
        response = agent.chat_response("I need help diagnosing a system issue")
        st.rerun()

with col2:
    if st.button("📋 System Status"):
        response = agent.chat_response("What's the current system status?")
        st.rerun()

with col3:
    if st.button("🆘 Emergency Help"):
        response = agent.chat_response("I need emergency assistance with a critical issue")
        st.rerun()
```

def show_diagnosis():
“”“Display issue diagnosis interface”””
st.header(“🔍 AI-Powered Issue Diagnosis”)

```
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Describe Your Issue")
    
    issue_description = st.text_area(
        "Issue Description",
        placeholder="e.g., Users are being logged out within a minute even with 'Remember Me' checked",
        height=150
    )
    
    error_code = st.text_input(
        "Error Code (if available)",
        placeholder="e.g., ERR1001"
    )
    
    affected_module = st.selectbox(
        "Affected Module",
        ["Auto-detect", "Login Service", "Transaction Engine", "Payment Gateway", 
         "Database Layer", "Core Banking", "Reporting Module", "Notification Service"]
    )
    
    if st.button("🔬 Diagnose Issue", type="primary"):
        if issue_description:
            diagnosis = agent.diagnose_issue(issue_description, error_code)
            st.session_state.current_diagnosis = diagnosis

with col2:
    st.subheader("🎯 Diagnosis Results")
    
    if 'current_diagnosis' in st.session_state:
        diagnosis = st.session_state.current_diagnosis
        
        # Display diagnosis
        if diagnosis["identified_module"]:
            st.markdown(f"""
            <div class="issue-card">
                <h4>🎯 Identified Module: {diagnosis["identified_module"]}</h4>
                <p><strong>Severity:</strong> {diagnosis["severity"].value}</p>
                <p><strong>Root Cause:</strong> {diagnosis["root_cause"]}</p>
                <p><strong>Estimated Resolution Time:</strong> {diagnosis["estimated_time"]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if diagnosis["escalation_needed"]:
                st.warning("⚠️ This issue requires escalation to senior support team!")
            
            # Resolution steps
            if diagnosis["resolution_steps"]:
                st.markdown("""
                <div class="resolution-card">
                    <h4>🛠️ Recommended Resolution Steps</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for i, step in enumerate(diagnosis["resolution_steps"], 1):
                    st.write(f"{i}. {step}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Mark as Resolved"):
                        st.success("Issue marked as resolved!")
                with col2:
                    if st.button("⬆️ Escalate Issue"):
                        st.warning("Issue escalated to senior team!")
        else:
            st.info("Please provide an issue description to get started.")
```

def show_report_issue():
“”“Display new issue reporting interface”””
st.header(“📝 Report New Maintenance Issue”)

```
with st.form("new_issue_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        title = st.text_input("Issue Title", placeholder="Brief description of the issue")
        module = st.selectbox("Affected Module", [
            "Login Service", "Transaction Engine", "Payment Gateway", 
            "Database Layer", "Core Banking", "Reporting Module", "Notification Service"
        ])
        severity = st.selectbox("Severity Level", [
            "Low", "Medium", "High", "Critical"
        ])
    
    with col2:
        error_code = st.text_input("Error Code (if available)", placeholder="e.g., ERR1001")
        reporter = st.text_input("Reporter Name", placeholder="Your name")
        affected_users = st.number_input("Estimated Affected Users", min_value=0, value=1)
    
    description = st.text_area(
        "Detailed Description", 
        placeholder="Provide detailed information about the issue, when it started, and any error messages",
        height=150
    )
    
    steps_to_reproduce = st.text_area(
        "Steps to Reproduce", 
        placeholder="1. Go to login page\n2. Enter credentials\n3. Click login\n4. Session expires immediately",
        height=100
    )
    
    submitted = st.form_submit_button("📤 Submit Issue Report", type="primary")
    
    if submitted:
        if title and description:
            # Create new issue
            new_issue = {
                "timestamp": datetime.now(),
                "title": title,
                "module": module,
                "description": description,
                "error_code": error_code,
                "severity": severity,
                "reporter": reporter,
                "affected_users": affected_users,
                "steps_to_reproduce": steps_to_reproduce,
                "status": "Open"
            }
            
            # Add to session state
            if 'reported_issues' not in st.session_state:
                st.session_state.reported_issues = []
            st.session_state.reported_issues.append(new_issue)
            
            st.success(f"✅ Issue reported successfully! Issue ID: ISSUE-{len(st.session_state.reported_issues):04d}")
            
            # Auto-diagnose the issue
            diagnosis = agent.diagnose_issue(description, error_code)
            
            st.markdown("### 🔬 Auto-Diagnosis Results")
            if diagnosis["identified_module"]:
                st.info(f"**Identified Module:** {diagnosis['identified_module']}")
                st.info(f"**Estimated Severity:** {diagnosis['severity'].value}")
                st.info(f"**Estimated Resolution Time:** {diagnosis['estimated_time']}")
                
                if diagnosis["resolution_steps"]:
                    st.markdown("**Suggested Resolution Steps:**")
                    for i, step in enumerate(diagnosis["resolution_steps"], 1):
                        st.write(f"{i}. {step}")
        else:
            st.error("Please fill in the required fields (Title and Description)")
```

def show_analytics(df):
“”“Display analytics and insights”””
st.header(“📈 Maintenance Analytics & Insights”)

```
# Time series analysis
st.subheader("📊 Issue Trends Over Time")

# Create hourly breakdown
df['hour'] = df['timestamp'].dt.hour
hourly_issues = df.groupby('hour').size().reset_index(name='count')

fig = px.line(hourly_issues, x='hour', y='count', 
              title="Issues by Hour of Day",
              labels={'hour': 'Hour', 'count': 'Number of Issues'})
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Module Performance")
    module_performance = df.groupby('module').agg({
        'status': lambda x: (x == 'Resolved').sum() / len(x) * 100
    }).round(2)
    module_performance.columns = ['Resolution Rate %']
    st.dataframe(module_performance, use_container_width=True)

with col2:
    st.subheader("⚡ Resolution Efficiency")
    # Simulated resolution times
    resolution_times = {
        'Login Service': '15 min',
        'Transaction Engine': '25 min', 
        'Payment Gateway': '35 min',
        'Database Layer': '45 min'
    }
    st.json(resolution_times)

# Advanced analytics
st.subheader("🧠 AI Insights")

insights = [
    "🔍 **Pattern Detected**: Login Service issues peak at 10 AM - likely due to morning login rush",
    "⚠️ **Alert**: Database Layer issues increased 40% this week - consider performance optimization",
    "✅ **Success**: Payment Gateway resolution time improved by 30% with new auto-retry mechanism",
    "📊 **Trend**: Transaction Engine deadlocks reduced after implementing query optimization",
]

for insight in insights:
    st.markdown(f"- {insight}")

# Recommendations
st.subheader("💡 AI Recommendations")

recommendations = [
    "Implement automated session extension for Login Service during peak hours",
    "Schedule database maintenance during low-traffic periods (2-4 AM)",
    "Add more redundancy to Payment Gateway connections",
    "Create automated alerts for issues that typically require escalation"
]

for i, rec in enumerate(recommendations, 1):
    st.write(f"{i}. {rec}")
```

if **name** == “**main**”:
main()
