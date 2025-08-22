import os
import pandas as pd
import tempfile
from typing import List, Dict, Any
from datetime import datetime
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import required libraries - corrected imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain.schema import Document
    from langchain.vectorstores import Chroma
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate
    from langchain.docstore.document import Document as LangDocument
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install the required packages:")
    print("pip install langchain langchain-google-genai chromadb pandas python-dotenv")
    exit(1)

class ApplicationMaintenanceResolver:
    def __init__(self, gemini_api_key: str = None, persist_directory: str = None):
        """
        Initialize the Application Maintenance Issue Resolver.
        """
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass it as argument.")
        
        self.persist_directory = persist_directory or "./chroma_db"
        
        # Initialize the LLM and embeddings
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.gemini_api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.gemini_api_key
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Initialize vector store
        self.vectorstore = None
        self.qa_chain = None
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def load_and_process_data(self, logs_csv_path: str = None, manual_md_path: str = None):
        """
        Load and process maintenance logs and troubleshooting manual.
        """
        # Use default paths if not provided
        logs_csv_path = logs_csv_path or "./data/maintenance_logs.csv"
        manual_md_path = manual_md_path or "./data/troubleshooting_manual.md"
        
        # Load maintenance logs
        logs_docs = self._load_logs(logs_csv_path)
        
        # Load troubleshooting manual
        manual_docs = self._load_manual(manual_md_path)
        
        # Combine all documents
        all_docs = logs_docs + manual_docs
        
        if not all_docs:
            print("Warning: No documents were loaded. Check your file paths.")
            # Create some dummy data for testing
            dummy_doc = LangDocument(
                page_content="Sample troubleshooting guide: Restart the application when encountering errors.",
                metadata={"source": "dummy", "type": "test"}
            )
            all_docs = [dummy_doc]
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(all_docs)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()
        
        # Create QA chain
        self._create_qa_chain()
        
        print(f"Loaded and processed {len(texts)} document chunks")
        return len(texts)
    
    def _load_logs(self, csv_path: str) -> List[LangDocument]:
        """
        Load maintenance logs from CSV file.
        """
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                print(f"Warning: Logs file not found at {csv_path}")
                return []
                
            df = pd.read_csv(csv_path)
            docs = []
            
            for _, row in df.iterrows():
                # Create a comprehensive text representation of the log entry
                content = f"""
                Maintenance Log Entry:
                Issue ID: {row.get('issue_id', 'N/A')}
                Timestamp: {row.get('timestamp', 'N/A')}
                Application: {row.get('application', 'N/A')}
                Module: {row.get('module', 'N/A')}
                Error Code: {row.get('error_code', 'N/A')}
                Issue Description: {row.get('issue_description', 'N/A')}
                Severity: {row.get('severity', 'N/A')}
                Resolution Steps: {row.get('resolution_steps', 'N/A')}
                Resolution Time: {row.get('resolution_time', 'N/A')}
                Technician: {row.get('technician', 'N/A')}
                Notes: {row.get('notes', 'N/A')}
                """
                
                metadata = {
                    "source": "maintenance_logs",
                    "issue_id": str(row.get('issue_id', '')),
                    "timestamp": str(row.get('timestamp', '')),
                    "application": str(row.get('application', '')),
                    "severity": str(row.get('severity', ''))
                }
                
                docs.append(LangDocument(page_content=content, metadata=metadata))
            
            print(f"Loaded {len(docs)} log entries from {csv_path}")
            return docs
        except Exception as e:
            print(f"Error loading logs from {csv_path}: {e}")
            return []
    
    def _load_manual(self, md_path: str) -> List[LangDocument]:
        """
        Load troubleshooting manual from markdown file.
        """
        try:
            # Check if file exists
            if not os.path.exists(md_path):
                print(f"Warning: Manual file not found at {md_path}")
                return []
                
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split the manual into sections based on headings
            sections = content.split('\n# ')
            docs = []
            
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                
                # Add back the # if it's not the first section
                if i > 0:
                    section = '# ' + section
                
                metadata = {
                    "source": "troubleshooting_manual",
                    "section_id": str(i),
                    "section_title": section.split('\n')[0].replace('#', '').strip()
                }
                
                docs.append(LangDocument(page_content=section, metadata=metadata))
            
            print(f"Loaded {len(docs)} manual sections from {md_path}")
            return docs
        except Exception as e:
            print(f"Error loading manual from {md_path}: {e}")
            return []
    
    def _create_qa_chain(self):
        """Create the conversational QA chain."""
        # Custom prompt template
        prompt_template = """You are an expert application maintenance issue resolver. 
        Use the following context to provide detailed, step-by-step resolution instructions for the given issue.

        Context:
        {context}

        Chat History:
        {chat_history}

        Question: {question}

        Please provide:
        1. A diagnosis of the issue
        2. Step-by-step resolution instructions
        3. Any additional notes or precautions

        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Reduced for better performance
        )
        
        # Create QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            output_key='answer'
        )
    
    def diagnose_issue(self, issue_description: str) -> Dict[str, Any]:
        """
        Diagnose an application maintenance issue and provide resolution steps.
        """
        if not self.qa_chain:
            raise ValueError("Please load data first using load_and_process_data()")
        
        # Create a detailed query
        query = f"""
        Application maintenance issue: {issue_description}
        
        Please provide:
        1. Diagnosis of the issue based on historical logs and manuals
        2. Step-by-step resolution instructions
        3. Estimated time for resolution
        4. Any potential risks or precautions
        """
        
        # Get response from the QA chain
        try:
            response = self.qa_chain({"question": query})
            
            # Extract source documents for reference
            source_docs = []
            for doc in response.get('source_documents', []):
                source_docs.append({
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata
                })
            
            return {
                "issue_id": str(uuid.uuid4())[:8],
                "timestamp": datetime.now().isoformat(),
                "issue_description": issue_description,
                "diagnosis": response['answer'],
                "source_documents": source_docs
            }
        except Exception as e:
            return {
                "issue_id": str(uuid.uuid4())[:8],
                "timestamp": datetime.now().isoformat(),
                "issue_description": issue_description,
                "diagnosis": f"Error processing request: {str(e)}",
                "source_documents": []
            }
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()

def main():
    """Main function to demonstrate the agent."""
    # Get API key from environment or user input
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Please enter your Gemini API key: ")
    
    # Initialize the resolver
    resolver = ApplicationMaintenanceResolver(api_key)
    
    # Try to load data from default paths
    try:
        doc_count = resolver.load_and_process_data()
        print(f"Successfully loaded {doc_count} document chunks")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using minimal knowledge base for demonstration...")
        # Create minimal knowledge base
        dummy_docs = [
            LangDocument(
                page_content="Common application issues: 1. Restart service 2. Check logs 3. Verify dependencies",
                metadata={"source": "default", "type": "general"}
            )
        ]
        texts = resolver.text_splitter.split_documents(dummy_docs)
        resolver.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=resolver.embeddings,
            persist_directory=resolver.persist_directory
        )
        resolver._create_qa_chain()
    
    # Test with sample issues
    sample_issues = [
        "Application crashes when processing large files",
        "Database connection timeout errors occurring frequently",
        "User authentication failing with error code 502"
    ]
    
    for issue in sample_issues:
        print(f"\n{'='*50}")
        print(f"Diagnosing issue: {issue}")
        print(f"{'='*50}")
        
        result = resolver.diagnose_issue(issue)
        
        print(f"Issue ID: {result['issue_id']}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"\nDiagnosis and Resolution:")
        print(result['diagnosis'])
        
        if result['source_documents']:
            print(f"\nReference Sources:")
            for i, doc in enumerate(result['source_documents'][:2]):  # Show first 2 sources
                source = doc['metadata'].get('source', 'Unknown')
                title = doc['metadata'].get('section_title', doc['metadata'].get('application', 'No title'))
                print(f"{i+1}. {source}: {title}")
        
        print(f"\n{'='*50}")

if __name__ == "__main__":
    main()