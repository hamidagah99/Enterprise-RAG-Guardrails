import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# NeMo Guardrails creates asyncio.Semaphore at import time, which requires a running
# event loop. The main thread has none by default on some platforms, so we set one first.
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from guardrail_actions import detect_sensitive_input, detect_sensitive_output

print("\nSelect backend:")
print("  [1] LM Studio  — local model (load your model in LM Studio first)")
print("  [2] University — KIConnect API (inferenz-gpt-oss-120b)")

choice = input("\nEnter 1 or 2: ").strip()
selected = "KIConnect (University)" if choice == "2" else "LM Studio"

if selected == "KIConnect (University)":
    llm = ChatOpenAI(
        base_url="https://chat.kiconnect.nrw/api/v1",
        model="inferenz-gpt-oss-120b",
        api_key=os.getenv("KICONNECT_API_KEY"),
        temperature=0,
    )
    nemo_dir = "./nemo_config_kiconnect"
else:
    llm = ChatOpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio", temperature=0)
    nemo_dir = "./nemo_config"

print(f"\nLoading documents and initialising {selected}...")

# reads every PDF inside the "documents" folder
loader = PyPDFDirectoryLoader("documents")
docs = loader.load()

if not docs:
    print("No PDFs found in the 'documents' folder. Add PDFs and restart.")
    exit()

print(f"Loaded {len(docs)} pages. Building vector database...")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# rewrites the user's question as a standalone query so previous chat turns don't confuse the retriever
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a chat history and the latest user question which might reference "
        "context in the chat history, formulate a standalone question that can be "
        "understood without the chat history. Do NOT answer — only reformulate."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

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
nemo_config = RailsConfig.from_path(nemo_dir)
protected_chain = RunnableRails(
    nemo_config,
    runnable=rag_chain,
    llm=llm,
    input_key="input",
    output_key="answer",
)

# register the keyword scanners so the Colang `execute` calls can find them
protected_chain.rails.register_action(detect_sensitive_input, name="detect_sensitive_input")
protected_chain.rails.register_action(detect_sensitive_output, name="detect_sensitive_output")

print(f"\nReady. Backend: {selected} | Guardrails: ON | Chat memory: ON")
print("Type 'quit' to exit.")
print("-" * 60)

chat_history = []

while True:
    user_input = input("\nYou: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ["quit", "exit"]:
        print("Session closed.")
        break

    result = protected_chain.invoke({"input": user_input, "chat_history": chat_history})
    answer = result.get("answer", "")

    print(f"\nAI: {answer}")

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=answer))
