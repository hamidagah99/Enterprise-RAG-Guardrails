import asyncio
import os

# NeMo Guardrails creates asyncio.Semaphore at import time, which requires a running
# event loop. Streamlit's ScriptRunner thread has none, so we set one before importing.
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from guardrail_actions import detect_sensitive_input, detect_sensitive_output

if not os.path.exists("documents"):
    os.makedirs("documents")

st.set_page_config(page_title="Local RAG AI", page_icon="🤖")
st.title("Secure Enterprise Knowledge Base")

with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.selected_model = st.selectbox(
        "Select Active Model:",
        ["Llama 3.1 (8B)", "Qwen 2.5 (7B)", "DeepSeek Coder V2"],
    )
    st.divider()
    st.header("📄 Upload Data")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Update Knowledge Base"):
        for file in uploaded_files:
            with open(os.path.join("documents", file.name), "wb") as f:
                f.write(file.getbuffer())
        st.cache_resource.clear()
        st.rerun()


@st.cache_resource(show_spinner=False)
def setup_ai():
    llm = ChatOpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio", temperature=0)

    loader = DirectoryLoader("documents", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    if not docs:
        return None

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # rewrites the user's question as a standalone query so chat history doesn't confuse the retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Given a chat history and the latest user question which might reference "
            "context in the chat history, formulate a standalone question that can be "
            "understood without the chat history. Do NOT answer — only reformulate."
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = (
        "You are a secure Enterprise Knowledge Assistant. "
        "You MUST ONLY use the retrieved context below to answer the question. "
        "NEVER draw on outside knowledge. "
        "If the answer is absent from the context, respond EXACTLY with: "
        "'I am restricted from answering this as the information is not in the "
        "provided enterprise documents.'"
        "\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        create_stuff_documents_chain(llm, qa_prompt),
    )

    # wraps the RAG chain so NeMo can intercept input and output before/after it runs
    nemo_config = RailsConfig.from_path("./nemo_config")
    protected_chain = RunnableRails(
        nemo_config,
        runnable=rag_chain,
        input_key="input",
        output_key="answer",
    )

    # register the keyword scanners so the Colang `execute` calls can find them
    protected_chain.rails.register_action(detect_sensitive_input, name="detect_sensitive_input")
    protected_chain.rails.register_action(detect_sensitive_output, name="detect_sensitive_output")

    return protected_chain


with st.spinner("Booting up AI and scanning documents..."):
    protected_chain = setup_ai()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if protected_chain is None:
    st.info("👈 Please upload a PDF in the sidebar to get started.")
else:
    if user_input := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # chat_history passes straight through to the RAG chain inside RunnableRails
                result = protected_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history,
                })
                # "answer" holds either the RAG response or the NeMo blocking message
                answer = result.get("answer", "")
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
