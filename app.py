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

from langchain_openai import ChatOpenAI
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

from guardrail_actions import detect_sensitive_input, detect_sensitive_output

DOCS_DIR = "documents"


def select_backend():
    print("\nSelect backend:")
    print("  [1] LM Studio  — local model ")
    print("  [2] University — BUW API ")

    choice = input("\nEnter 1 or 2: ").strip()
    selected = "BUW API" if choice == "2" else "LM Studio"

    if selected == "BUW API":
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

    return llm, nemo_dir, selected


def load_retrieval_chain(llm):
    """Optional add-on: if the documents/ folder has files dropped into it, build a
    retrieval-augmented chain over them. Returns None (no retrieval) if the folder is
    missing or empty — this must never raise or exit, since running without any
    documents is the default, fully-supported mode.
    """
    if not os.path.isdir(DOCS_DIR):
        return None

    has_files = any(
        os.path.isfile(os.path.join(DOCS_DIR, name)) for name in os.listdir(DOCS_DIR)
    )
    if not has_files:
        return None

    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = PyPDFDirectoryLoader(DOCS_DIR).load()
    if not docs:
        return None

    print(f"Loaded {len(docs)} pages from '{DOCS_DIR}/'. Building vector index...")

    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    retriever = Chroma.from_documents(documents=splits, embedding=embeddings).as_retriever()

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

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Answer the question using the retrieved context below. If the answer isn't "
            "in the context, say you don't know.\n\n{context}"
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    return create_retrieval_chain(history_aware_retriever, create_stuff_documents_chain(llm, qa_prompt))


def build_protected_llm(llm, nemo_dir: str, runnable=None) -> RunnableRails:
    """Wraps a chat LLM (optionally a retrieval chain) with NeMo Guardrails input/output
    rails. The rails are identical either way — retrieval is an optional add-on, not a
    dependency of the guardrail behavior.
    """
    nemo_config = RailsConfig.from_path(nemo_dir)

    if runnable is not None:
        protected_llm = RunnableRails(
            nemo_config, runnable=runnable, llm=llm, input_key="input", output_key="answer"
        )
    else:
        protected_llm = RunnableRails(nemo_config, llm=llm)

    protected_llm.rails.register_action(detect_sensitive_input, name="detect_sensitive_input")
    protected_llm.rails.register_action(detect_sensitive_output, name="detect_sensitive_output")

    return protected_llm


def ask(protected_llm: RunnableRails, chat_history: list, user_input: str) -> str:
    if protected_llm.passthrough_runnable is not None:
        result = protected_llm.invoke({"input": user_input, "chat_history": chat_history})
        return result.get("answer", "")

    messages = chat_history + [{"role": "user", "content": user_input}]
    result = protected_llm.invoke({"input": messages})

    output = result.get("output") if isinstance(result, dict) else result
    if isinstance(output, dict):
        return output.get("content", "")
    return str(output)


def run_chat_loop(protected_llm: RunnableRails, backend_name: str, retrieval_on: bool) -> None:
    mode = "Retrieval: ON (documents/)" if retrieval_on else "Retrieval: OFF (no documents)"
    print(f"\nReady. Backend: {backend_name} | Guardrails: ON | {mode}")
    print("Type 'quit' to exit.")
    print("-" * 60)

    chat_history: list = []

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Session closed.")
            break

        answer = ask(protected_llm, chat_history, user_input)
        print(f"\nAI: {answer}")

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    llm, nemo_dir, backend_name = select_backend()
    print(f"\nInitialising {backend_name}...")

    retrieval_chain = load_retrieval_chain(llm)
    if retrieval_chain is None:
        print(f"No documents found in '{DOCS_DIR}/' — running as a plain guarded chat.")

    protected_llm = build_protected_llm(llm, nemo_dir, runnable=retrieval_chain)
    run_chat_loop(protected_llm, backend_name, retrieval_on=retrieval_chain is not None)
