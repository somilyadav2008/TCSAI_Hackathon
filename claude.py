import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
from typing import List, Dict, Any
import os

# LangChain imports
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory

# Configure Streamlit page
st.set_page_config(
    page_title="Application Maintenance Issue Resolver",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .resolution-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    .success-box {
        background-color: #d1edff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

class MaintenanceIssueResolver:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.agent = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
    def initialize_models(self):
        """Initialize the local LLM and embeddings"""
        try:
            # Initialize embeddings with GTE-Large
            self.embeddings = OllamaEmbeddings(
                model="gte-large",
                base_url="http://localhost:11434"
            )
            
            # Initialize LLM with Ollama
            self.llm = Ollama(
                model="llama3.1",  # or your preferred model
                base_url="http://localhost:11434",
                temperature=0.1
            )
            
            return True
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            return False
    
    def process_log_file(self, log_content: str) -> List[Document]:
        """Process maintenance logs into documents"""
        documents = []
        
        # Split log content by common log patterns
        log_entries = re.split(r'\n(?=\d{4}-\d{2}-\d{2}|\[\d{4}-\d{2}-\d{2})', log_content)
        
        for i, entry in enumerate(log_entries):
            if entry.strip():
                # Extract timestamp, severity, and message
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', entry)
                severity_match = re.search(r'(ERROR|WARN|INFO|DEBUG|FATAL)', entry, re.IGNORECASE)
                
                timestamp = timestamp_match.group(1) if timestamp_match else f"entry_{i}"
                severity = severity_match.group(1) if severity_match else "INFO"
                
                doc = Document(
                    page_content=entry.strip(),
                    metadata={
                        "source": "maintenance_logs",
                        "timestamp": timestamp,
                        "severity": severity,
                        "entry_id": i
                    }
                )
                documents.append(doc)
        
        return documents
    
    def process_manual_file(self, manual_content: str) -> List[Document]:
        """Process troubleshooting manual into documents"""
        documents = []
        
        # Split manual into sections
        sections = re.split(r'\n(?=#+\s+|\d+\.\s+|[A-Z][A-Z\s]+:)', manual_content)
        
        for i, section in enumerate(sections):
            if section.strip() and len(section.strip()) > 50:
                # Extract section title
                lines = section.strip().split('\n')
                title = lines[0].strip() if lines else f"Section {i}"
                
                doc = Document(
                    page_content=section.strip(),
                    metadata={
                        "source": "troubleshooting_manual",
                        "section": title,
                        "section_id": i
                    }
                )
                documents.append(doc)
        
        return documents
    
    def create_vectorstore(self, documents: List[Document]):
        """Create FAISS vectorstore from documents"""
        if not documents:
            st.error("No documents to process")
            return False
        
        try:
            # Split documents into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vectorstore
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            return True
        except Exception as e:
            st.error(f"Error creating vectorstore: {str(e)}")
            return False
    
    def create_qa_chain(self):
        """Create QA chain with custom prompt"""
        if not self.vectorstore or not self.llm:
            return False
        
        try:
            # Custom prompt for maintenance issue resolution
            prompt_template = """
            You are an expert application maintenance engineer specializing in banking systems. 
            Use the following context from maintenance logs and troubleshooting guides to provide 
            detailed, step-by-step resolution instructions for the given issue.
            
            Context: {context}
            
            Question: {question}
            
            Provide your response in the following format:
            
            **ISSUE DIAGNOSIS:**
            - Identify the root cause
            - Assess the severity level
            - Determine affected systems/components
            
            **RESOLUTION STEPS:**
            1. [Step-by-step instructions]
            2. [Include specific commands, configurations, or procedures]
            3. [Mention any prerequisites or dependencies]
            
            **VALIDATION:**
            - How to verify the fix worked
            - What to monitor post-resolution
            
            **PREVENTION:**
            - Recommendations to prevent similar issues
            
            If you cannot find specific information in the context, clearly state what additional 
            information would be needed and provide general best practices for similar issues.
            
            Answer:
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            return True
        except Exception as e:
            st.error(f"Error creating QA chain: {str(e)}")
            return False
    
    def create_diagnostic_tools(self):
        """Create tools for the diagnostic agent"""
        
        def search_logs_tool(query: str) -> str:
            """Search maintenance logs for relevant information"""
            if not self.vectorstore:
                return "Vectorstore not available"
            
            docs = self.vectorstore.similarity_search(query, k=3)
            results = []
            for doc in docs:
                if doc.metadata.get("source") == "maintenance_logs":
                    results.append(f"Log Entry: {doc.page_content[:200]}...")
            
            return "\n".join(results) if results else "No relevant log entries found"
        
        def search_manual_tool(query: str) -> str:
            """Search troubleshooting manual for relevant procedures"""
            if not self.vectorstore:
                return "Vectorstore not available"
            
            docs = self.vectorstore.similarity_search(query, k=3)
            results = []
            for doc in docs:
                if doc.metadata.get("source") == "troubleshooting_manual":
                    results.append(f"Manual Section: {doc.page_content[:200]}...")
            
            return "\n".join(results) if results else "No relevant manual sections found"
        
        def issue_categorizer_tool(issue_description: str) -> str:
            """Categorize the issue type"""
            categories = {
                "performance": ["slow", "timeout", "latency", "response time"],
                "database": ["sql", "database", "connection", "query"],
                "authentication": ["login", "auth", "password", "token"],
                "network": ["network", "connection", "port", "firewall"],
                "application": ["crash", "error", "exception", "bug"],
                "security": ["security", "breach", "vulnerability", "unauthorized"]
            }
            
            issue_lower = issue_description.lower()
            detected_categories = []
            
            for category, keywords in categories.items():
                if any(keyword in issue_lower for keyword in keywords):
                    detected_categories.append(category)
            
            return f"Detected categories: {', '.join(detected_categories) if detected_categories else 'general'}"
        
        return [
            Tool(
                name="search_logs",
                description="Search maintenance logs for relevant error messages and historical issues",
                func=search_logs_tool
            ),
            Tool(
                name="search_manual",
                description="Search troubleshooting manual for relevant procedures and solutions",
                func=search_manual_tool
            ),
            Tool(
                name="categorize_issue",
                description="Categorize the type of maintenance issue",
                func=issue_categorizer_tool
            )
        ]
    
    def resolve_issue(self, issue_description: str) -> Dict[str, Any]:
        """Main method to resolve maintenance issues"""
        if not self.qa_chain:
            return {"error": "System not properly initialized"}
        
        try:
            # Get response from QA chain
            response = self.qa_chain({"query": issue_description})
            
            # Extract source documents for reference
            source_docs = response.get("source_documents", [])
            sources = []
            for doc in source_docs:
                sources.append({
                    "source": doc.metadata.get("source", "unknown"),
                    "content": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                })
            
            return {
                "resolution": response["result"],
                "sources": sources,
                "confidence": "high" if len(source_docs) >= 3 else "medium"
            }
        
        except Exception as e:
            return {"error": f"Error resolving issue: {str(e)}"}

# Initialize the resolver
@st.cache_resource
def get_resolver():
    return MaintenanceIssueResolver()

def create_sample_data():
    """Create sample maintenance logs and manual data"""
    sample_logs = """
2024-01-15 08:30:15 ERROR BankingApp Database connection timeout after 30 seconds
2024-01-15 08:30:16 WARN BankingApp Retrying database connection (attempt 1/3)
2024-01-15 08:30:45 ERROR BankingApp Database connection failed - all retry attempts exhausted
2024-01-15 08:31:00 INFO BankingApp Switching to backup database server
2024-01-15 08:31:02 INFO BankingApp Connection restored to backup server

2024-01-16 14:22:33 ERROR AuthService Invalid JWT token received from client
2024-01-16 14:22:33 WARN AuthService Client IP 192.168.1.100 sending malformed tokens
2024-01-16 14:22:45 INFO AuthService Rate limiting applied to IP 192.168.1.100

2024-01-17 09:15:20 ERROR TransactionService Transaction processing failed - insufficient funds
2024-01-17 09:15:21 INFO TransactionService Account balance check completed for account #12345
2024-01-17 09:15:22 INFO TransactionService Transaction declined - balance insufficient

2024-01-18 11:30:00 FATAL BankingApp OutOfMemoryError in transaction processing module
2024-01-18 11:30:01 ERROR BankingApp Application crash detected
2024-01-18 11:30:05 INFO BankingApp Initiating automatic restart sequence
2024-01-18 11:31:00 INFO BankingApp Application restarted successfully

2024-01-19 16:45:10 ERROR APIGateway HTTP 503 Service Unavailable - downstream service timeout
2024-01-19 16:45:11 WARN APIGateway Circuit breaker opened for payment service
2024-01-19 16:45:30 INFO APIGateway Circuit breaker closed - service recovered
"""

    sample_manual = """
# BANKING APPLICATION TROUBLESHOOTING MANUAL

## 1. DATABASE CONNECTION ISSUES

### 1.1 Connection Timeout Errors
**Symptoms:** Database connection timeout after 30 seconds
**Root Cause:** Network latency or database server overload
**Resolution Steps:**
1. Check database server status: `systemctl status postgresql`
2. Verify network connectivity: `ping database-server`
3. Check connection pool settings in application.properties
4. Restart application service if needed: `systemctl restart banking-app`
5. Monitor connection pool metrics

### 1.2 Connection Pool Exhaustion
**Symptoms:** Unable to obtain database connection from pool
**Resolution Steps:**
1. Check current pool size: `SELECT count(*) FROM pg_stat_activity;`
2. Increase pool size in configuration
3. Identify long-running queries and optimize
4. Restart application to reset pool

## 2. AUTHENTICATION SERVICE ISSUES

### 2.1 JWT Token Validation Errors
**Symptoms:** Invalid JWT token errors in logs
**Root Cause:** Token expiry or signature verification failure
**Resolution Steps:**
1. Check token expiration settings
2. Verify JWT secret key configuration
3. Restart AuthService: `systemctl restart auth-service`
4. Clear client-side token cache
5. Monitor token generation and validation rates

### 2.2 Rate Limiting Issues
**Symptoms:** Clients receiving 429 Too Many Requests
**Resolution Steps:**
1. Review rate limiting configuration
2. Check if legitimate traffic is being blocked
3. Adjust rate limits if necessary
4. Implement IP whitelisting for trusted sources

## 3. TRANSACTION PROCESSING ISSUES

### 3.1 Insufficient Funds Errors
**Symptoms:** Transaction failures due to insufficient balance
**Resolution Steps:**
1. Verify account balance calculation logic
2. Check for pending transactions affecting balance
3. Review overdraft protection settings
4. Ensure real-time balance updates

### 3.2 Transaction Timeout Issues
**Symptoms:** Transactions timing out during processing
**Resolution Steps:**
1. Check database query performance
2. Review transaction isolation levels
3. Optimize database indexes
4. Increase transaction timeout settings

## 4. MEMORY AND PERFORMANCE ISSUES

### 4.1 OutOfMemoryError
**Symptoms:** Application crashes with OutOfMemoryError
**Root Cause:** Memory leaks or insufficient heap size
**Resolution Steps:**
1. Analyze heap dump: `jmap -dump:live,format=b,file=heapdump.hprof [PID]`
2. Increase JVM heap size: `-Xmx4g -Xms2g`
3. Review code for memory leaks
4. Implement proper garbage collection tuning
5. Monitor memory usage patterns

### 4.2 High CPU Usage
**Symptoms:** CPU usage consistently above 80%
**Resolution Steps:**
1. Identify CPU-intensive processes: `top -p [PID]`
2. Analyze thread dumps
3. Review inefficient algorithms
4. Scale horizontally if needed

## 5. API GATEWAY ISSUES

### 5.1 Service Unavailable (503) Errors
**Symptoms:** Downstream services returning 503 errors
**Resolution Steps:**
1. Check downstream service health
2. Review circuit breaker configuration
3. Verify load balancer settings
4. Implement proper retry mechanisms
5. Monitor service dependencies

### 5.2 Circuit Breaker Activation
**Symptoms:** Circuit breaker opens frequently
**Resolution Steps:**
1. Review failure threshold settings
2. Check downstream service performance
3. Implement proper fallback mechanisms
4. Adjust timeout values
5. Monitor error rates and response times

## 6. MONITORING AND ALERTING

### 6.1 Setting Up Alerts
**Key Metrics to Monitor:**
- Database connection pool utilization
- API response times
- Error rates by service
- Memory and CPU usage
- Transaction success rates

### 6.2 Log Analysis
**Important Log Patterns:**
- ERROR: Critical issues requiring immediate attention
- WARN: Potential issues to investigate
- Connection timeouts: Network or performance issues
- Authentication failures: Security concerns
"""

    return sample_logs, sample_manual

def main():
    st.markdown('<h1 class="main-header">🔧 Application Maintenance Issue Resolver</h1>', unsafe_allow_html=True)
    
    # Initialize resolver
    resolver = get_resolver()
    
    # Sidebar for configuration and file uploads
    with st.sidebar:
        st.markdown('<h2 class="section-header">📁 Data Sources</h2>', unsafe_allow_html=True)
        
        # Option to use sample data or upload files
        data_source = st.radio(
            "Choose data source:",
            ["Use Sample Data (Banking App)", "Upload Custom Files"]
        )
        
        if data_source == "Use Sample Data (Banking App)":
            if st.button("Load Sample Data"):
                with st.spinner("Loading sample data..."):
                    sample_logs, sample_manual = create_sample_data()
                    st.session_state['logs_content'] = sample_logs
                    st.session_state['manual_content'] = sample_manual
                    st.success("Sample data loaded successfully!")
        
        else:
            # File upload section
            uploaded_logs = st.file_uploader(
                "Upload Maintenance Logs",
                type=['txt', 'log'],
                help="Upload your application maintenance logs"
            )
            
            uploaded_manual = st.file_uploader(
                "Upload Troubleshooting Manual",
                type=['txt', 'md'],
                help="Upload your troubleshooting manual or procedures"
            )
            
            if uploaded_logs and uploaded_manual:
                with st.spinner("Processing uploaded files..."):
                    logs_content = str(uploaded_logs.read(), "utf-8")
                    manual_content = str(uploaded_manual.read(), "utf-8")
                    st.session_state['logs_content'] = logs_content
                    st.session_state['manual_content'] = manual_content
                    st.success("Files uploaded successfully!")
        
        # Model initialization section
        st.markdown('<h2 class="section-header">🤖 Model Setup</h2>', unsafe_allow_html=True)
        
        if st.button("Initialize Models & Process Data"):
            if 'logs_content' not in st.session_state or 'manual_content' not in st.session_state:
                st.error("Please load data first!")
                return
            
            with st.spinner("Initializing models and processing data..."):
                # Initialize models
                if not resolver.initialize_models():
                    st.error("Failed to initialize models. Make sure Ollama is running!")
                    return
                
                # Process documents
                log_docs = resolver.process_log_file(st.session_state['logs_content'])
                manual_docs = resolver.process_manual_file(st.session_state['manual_content'])
                all_docs = log_docs + manual_docs
                
                # Create vectorstore
                if not resolver.create_vectorstore(all_docs):
                    st.error("Failed to create vectorstore!")
                    return
                
                # Create QA chain
                if not resolver.create_qa_chain():
                    st.error("Failed to create QA chain!")
                    return
                
                st.session_state['system_initialized'] = True
                st.success(f"System initialized! Processed {len(log_docs)} log entries and {len(manual_docs)} manual sections.")
    
    # Main content area
    if 'system_initialized' not in st.session_state:
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ System Not Ready</h3>
            <p>Please follow these steps to get started:</p>
            <ol>
                <li>Load sample data or upload your own files using the sidebar</li>
                <li>Click "Initialize Models & Process Data"</li>
                <li>Wait for the system to process your data</li>
            </ol>
            <p><strong>Note:</strong> Make sure Ollama is running locally with the required models (llama3.1 and gte-large)</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Issue resolution interface
    st.markdown('<h2 class="section-header">🔍 Issue Resolution</h2>', unsafe_allow_html=True)
    
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["🚨 Resolve Issue", "📊 Sample Scenarios", "📈 Analytics"])
    
    with tab1:
        # Issue description input
        st.markdown("### Describe the Issue")
        issue_description = st.text_area(
            "Enter the maintenance issue description:",
            height=100,
            placeholder="Example: Database connection timeout errors occurring frequently in the banking application..."
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            resolve_button = st.button("🔧 Resolve Issue", type="primary")
        
        with col2:
            urgency = st.selectbox("Urgency Level", ["Low", "Medium", "High", "Critical"])
        
        if resolve_button and issue_description:
            with st.spinner("Analyzing issue and generating resolution..."):
                result = resolver.resolve_issue(issue_description)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Display resolution
                    st.markdown('<div class="resolution-box">', unsafe_allow_html=True)
                    st.markdown("### 🎯 Resolution")
                    st.markdown(result["resolution"])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display sources
                    if result.get("sources"):
                        with st.expander("📚 Reference Sources"):
                            for i, source in enumerate(result["sources"]):
                                st.write(f"**Source {i+1}** ({source['source']}):")
                                st.write(source["content"])
                                st.divider()
                    
                    # Display confidence level
                    confidence_color = "🟢" if result.get("confidence") == "high" else "🟡"
                    st.markdown(f"**Confidence Level:** {confidence_color} {result.get('confidence', 'medium').title()}")
    
    with tab2:
        st.markdown("### 📋 Sample Maintenance Scenarios")
        
        sample_scenarios = [
            "Database connection timeout errors occurring frequently",
            "JWT authentication tokens are being rejected",
            "Application crashing with OutOfMemoryError",
            "API gateway returning 503 Service Unavailable",
            "Transaction processing is slow and timing out",
            "High CPU usage causing performance degradation"
        ]
        
        for i, scenario in enumerate(sample_scenarios):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.write(f"**{i+1}.** {scenario}")
            
            with col2:
                if st.button(f"Resolve", key=f"scenario_{i}"):
                    with st.spinner("Resolving..."):
                        result = resolver.resolve_issue(scenario)
                        if "error" not in result:
                            st.session_state[f'resolution_{i}'] = result
            
            # Display resolution if available
            if f'resolution_{i}' in st.session_state:
                with st.expander(f"Resolution for Scenario {i+1}", expanded=True):
                    st.markdown(st.session_state[f'resolution_{i}']['resolution'])
    
    with tab3:
        st.markdown("### 📊 System Analytics")
        
        if 'logs_content' in st.session_state:
            # Parse logs for basic analytics
            logs = st.session_state['logs_content']
            
            # Count error types
            error_counts = {
                'ERROR': len(re.findall(r'ERROR', logs)),
                'WARN': len(re.findall(r'WARN', logs)),
                'INFO': len(re.findall(r'INFO', logs)),
                'FATAL': len(re.findall(r'FATAL', logs))
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Log Level Distribution")
                st.bar_chart(error_counts)
            
            with col2:
                st.markdown("#### System Components")
                components = re.findall(r'(BankingApp|AuthService|TransactionService|APIGateway)', logs)
                component_counts = {comp: components.count(comp) for comp in set(components)}
                if component_counts:
                    st.bar_chart(component_counts)
            
            # Recent issues timeline
            st.markdown("#### Recent Issues Timeline")
            dates = re.findall(r'(\d{4}-\d{2}-\d{2})', logs)
            if dates:
                date_counts = {date: dates.count(date) for date in set(dates)}
                st.line_chart(date_counts)

if __name__ == "__main__":
    main()