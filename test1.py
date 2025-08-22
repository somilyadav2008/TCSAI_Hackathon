# src/agent.py
from typing import List
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory

from config import settings
from llm import make_llm
from tools import TOOLS


def make_memory():
    # Long-term memory (FAISS)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    try:
        vs = FAISS.load_local(
            settings.memory_store_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception:
        vs = FAISS.from_texts(["System memory initialized."], embedding=embeddings)
        vs.save_local(settings.memory_store_dir)

    retriever = vs.as_retriever(search_kwargs={"k": 4})

    # NOTE: Deprecation warnings are expected with pinned deps
    long_mem = VectorStoreRetrieverMemory(
        retriever=retriever,
        memory_key="history",
    )
    short_mem = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return long_mem, short_mem, vs


def make_agent():
    llm = make_llm()
    long_mem, short_mem, memory_store = make_memory()

    # Prepare tool descriptions
    tool_desc = "\n".join(
        f"- {t.name}: {t.description or 'No description'}" for t in TOOLS
    )
    tool_names = ", ".join(t.name for t in TOOLS)

    # ReAct prompt:
    # - {tools}, {tool_names}
    # - chat_history as messages
    # - agent_scratchpad as STRING (important for LangChain 0.2.x ReAct agent)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a senior SRE/Support engineer for a Banking Management System.\n"
                    "Use tools when helpful. Prefer citing error codes and stepwise runbooks.\n"
                    "If the user reports a symptom, retrieve related KB and propose diagnostics & resolution.\n\n"
                    "Available tools:\n{tools}\n\n"
                    "Call tools only by these names: {tool_names}\n"
                ),
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            # IMPORTANT: keep agent_scratchpad as a STRING placeholder for this LC version
            ("ai", "{agent_scratchpad}"),
        ]
    ).partial(tools=tool_desc, tool_names=tool_names)

    # Build agent and executor
    agent = create_react_agent(llm=llm, tools=TOOLS, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        memory=short_mem,  # short-term memory for chat history
        handle_parsing_errors=True,
        max_iterations=5
    )

    class AgentWithMemory:
        def __init__(self, exec_, long_term_mem, vs_):
            self.exec = exec_
            self.long_mem = long_term_mem
            self.vs = vs_

        def aask(self, query: str):
            # Do NOT pass agent_scratchpad; AgentExecutor will supply a string
            return self.exec.invoke({"input": query})

        def remember(self, note: str):
            self.vs.add_texts([note])
            self.vs.save_local(settings.memory_store_dir)
            return "Noted."

    return AgentWithMemory(executor, long_mem, memory_store)


if __name__ == "__main__":
    agent = make_agent()
    print(agent.aask("Hello, what can you do?"))

-------------------------------------------------------------------------------------------------


import os
import shutil
import streamlit as st

from config import settings
from agent import make_agent

st.set_page_config(page_title="Banking Maintenance RAG Agent", page_icon="🛠️", layout="wide")

# ---------- Helpers ----------
def init_session():
    if "agent" not in st.session_state:
        st.session_state.agent = make_agent()
    if "chat" not in st.session_state:
        st.session_state.chat = []  # list of dicts: {"role": "user"/"assistant", "content": str}

def reset_short_term():
    # Recreate the agent to clear ConversationBufferMemory
    st.session_state.agent = make_agent()
    st.session_state.chat = []
    st.toast("Short-term memory cleared (session reset).", icon="✅")

def clear_vector_memory():
    # Danger: wipes long-term FAISS store
    mem_dir = settings.memory_store_dir
    try:
        if os.path.exists(mem_dir):
            shutil.rmtree(mem_dir)
        # rebuild empty agent vector store
        st.session_state.agent = make_agent()
        st.toast(f"Long-term vector memory cleared: {mem_dir}", icon="🧹")
    except Exception as e:
        st.error(f"Failed to clear vector memory: {e}")

def add_texts_to_memory(texts):
    try:
        ag = st.session_state.agent
        # AgentWithMemory in agent.py keeps 'vs' accessible
        ag.vs.add_texts(texts)
        ag.vs.save_local(settings.memory_store_dir)
        st.toast(f"Added {len(texts)} chunk(s) to long-term memory.", icon="📚")
    except Exception as e:
        st.error(f"Failed to add to memory: {e}")

def vector_count():
    try:
        vs = st.session_state.agent.vs
        # FAISS vectorstore exposes index.ntotal
        return int(getattr(vs.index, "ntotal", 0))
    except Exception:
        return 0

# ---------- UI ----------
init_session()

st.title("🛠️ Banking Maintenance RAG Agent")
st.caption("LangChain • ReAct • Tools • Short-term + FAISS long-term memory • Local LLM via Ollama")

with st.sidebar:
    st.subheader("⚙️ Runtime")
    st.write(f"**Provider**: `{settings.provider}`")
    if settings.provider.lower() == "ollama":
        st.write(f"**Model**: `{settings.ollama_model}`")
    elif settings.provider.lower() == "openai":
        st.write(f"**OpenAI Model**: `{settings.openai_model}`")
    st.write(f"**Embedding model**: `{settings.embedding_model}`")
    st.write(f"**Memory dir**: `{settings.memory_store_dir}`")
    st.write(f"**Vector memory size**: `{vector_count()}`")

    st.divider()
    st.subheader("🧠 Memory Controls")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Reset short-term", use_container_width=True):
            reset_short_term()
    with colB:
        if st.button("Clear vector memory", type="primary", use_container_width=True):
            clear_vector_memory()

    st.divider()
    st.subheader("📥 Upload snippets into long-term memory")
    up = st.file_uploader(
        "Upload TXT/LOG/MD/CSV to store as retrievable memory chunks",
        type=["txt", "log", "md", "csv"],
        accept_multiple_files=True
    )
    if up:
        new_texts = []
        for f in up:
            try:
                content = f.read().decode("utf-8", errors="ignore")
                new_texts.append(f"[{f.name}] {content}")
            except Exception as e:
                st.warning(f"{f.name}: {e}")
        if new_texts and st.button(f"Add {len(new_texts)} file(s) to memory"):
            add_texts_to_memory(new_texts)

    st.divider()
    st.subheader("📝 Quick Remember")
    memo = st.text_area("Store a fact in long-term memory (FAISS):", placeholder="Example: UAT DB reset requires DBA approval.")
    if st.button("Remember"):
        if memo.strip():
            st.session_state.agent.remember(memo.strip())
            st.success("Noted in FAISS memory.")
        else:
            st.info("Enter something to remember.")

# Main chat layout
chat_container = st.container()

with chat_container:
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask about incidents, errors, logs…")
    if user_prompt:
        st.session_state.chat.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Call the agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    out = st.session_state.agent.aask(user_prompt)
                    # The AgentExecutor returns {"output": "...", ...}
                    # In verbose chains, it may also print tool logs to console.
                    answer = out.get("output", str(out))
                    st.markdown(answer)
                    st.session_state.chat.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Agent error: {e}")



("system",
 (
     "You are a senior SRE/Support engineer for a Banking Management System.\n"
     "Use tools when helpful. Prefer citing error codes and stepwise runbooks.\n"
     "If the user reports a symptom, retrieve related KB and propose diagnostics & resolution.\n\n"
     "Available tools:\n{tools}\n\n"
     "Call tools only by these names: {tool_names}\n\n"
     "When reasoning or solving a task, follow this format:\n"
     "Thought: <reasoning about the next step>\n"
     "Action: <tool name from the list above>\n"
     "Action Input: <string input to the tool>\n"
     "Observation: <tool output will appear here>\n"
     "Repeat Thought/Action... until final answer.\n"
     "When done, respond with:\n"
     "Final Answer: <your answer to the user>\n"
 ),
),





