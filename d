import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

# ----------- Set up once on app start --------------
@st.cache_resource
def setup_pipeline():
    # Load CSV
    loader = CSVLoader(file_path=r'U:\Cipher\react_app\employees.csv')
    documents = loader.load()

    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Load embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vector store
    vectorstore = FAISS.from_documents(docs, embedding_model)

    # Load LLM via Ollama
    llm = Ollama(model="llama-3.2-3b-it:latest", temperature=0.7)

    return vectorstore, llm

# --------------- Streamlit UI ----------------------
st.set_page_config(page_title="Employee Q&A", layout="wide")
st.title("🔍 Employee Search using LLaMA 3 + FAISS")

# Set up backend pipeline
vectorstore, llm = setup_pipeline()

# User input
query = st.text_input("Ask a question about employees:", placeholder="e.g., Who works in the Engineering department?")

if query:
    # Retrieve relevant documents
    relevant_docs = vectorstore.similarity_search(query, k=3)

    # Create context
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Construct prompt
    final_prompt = f"""Use the following employee data to answer the question.

{context}

Question: {query}
Answer:"""

    # Get LLM response
    response = llm.invoke(final_prompt)

    # Display response
    st.markdown("### 🧠 Answer")
    st.success(response)
