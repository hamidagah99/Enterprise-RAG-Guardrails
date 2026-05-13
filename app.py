import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

if not os.path.exists("documents"):
    os.makedirs("documents")

class GuardrailManager:
    def __init__(self):
        self.hr_pii_keywords = ["salary", "salaries", "pay", "gehalt", "lohn", "bonus", "telefonnummer", "adresse"]
        self.it_sec_keywords = ["password", "api key", "passwort", "zugangscode", "rfid", "token"]

    def get_system_prompt(self):
        return """You are a secure Enterprise Knowledge Assistant. 
You must strictly obey the following rules:

1. NO OUTSIDE KNOWLEDGE: You may ONLY use the information provided in the Context below to answer the user's question.
2. REFUSAL PROTOCOL: If the answer is not explicitly contained in the Context, you MUST reply exactly with: "I am restricted from answering this as the information is not in the provided enterprise documents."
3. CATEGORY 1 (HR/PII): If the user asks about personal employee data or financial compensation, you MUST refuse to answer.
4. CATEGORY 2 (RESEARCH/IP): If the user asks about unpublished research, trade secrets, or proprietary algorithms, you MUST reply exactly with: "I am restricted from discussing unreleased research or intellectual property."

Context: {context}"""

    def check_input(self, text):
        text_lower = text.lower()
        if any(word in text_lower for word in self.it_sec_keywords):
            return True, "🚨 CATEGORY 3 (IT SECURITY) TRIGGERED: Request contains restricted system access keywords. Blocked before reaching AI."
        if any(word in text_lower for word in self.hr_pii_keywords):
            return True, "🚨 CATEGORY 1 (HR/PII) TRIGGERED: Request contains restricted financial/personal keywords. Blocked before reaching AI."
        return False, ""

    def check_output(self, text):
        text_lower = text.lower()
        if any(word in text_lower for word in self.hr_pii_keywords):
            return True, "🚨 CATEGORY 1 (HR/PII) OUTPUT BLOCKED: The AI attempted to display restricted financial/personal data. Response intercepted."
        if any(word in text_lower for word in self.it_sec_keywords):
            return True, "🚨 CATEGORY 3 (IT SECURITY) OUTPUT BLOCKED: The AI attempted to display restricted system credentials. Response intercepted."
        return False, ""

guardrails = GuardrailManager()

st.set_page_config(page_title="Local RAG AI", page_icon="🤖")
st.title("Secure Enterprise Knowledge Base")

with st.sidebar:
    st.header("⚙️ Thesis Settings")
    st.session_state.selected_model = st.selectbox(
        "Select Active Model:",
        ["Llama 3.1 (8B)", "Qwen 2.5 (7B)", "DeepSeek Coder V2"]
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
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", guardrails.get_system_prompt()),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
   
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

with st.spinner("Booting up AI and scanning documents..."):
    rag_chain = setup_ai()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if rag_chain is None:
    st.info("👈 Please upload a PDF in the sidebar to get started.")
else:
    if user_input := st.chat_input("Ask a question about your documents..."):
        
        is_blocked, block_message = guardrails.check_input(user_input)
        
        if is_blocked:
            st.error(block_message)
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": block_message})
            st.rerun()

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})
                answer = response["answer"]
                
                is_out_blocked, out_block_message = guardrails.check_output(answer)
                
                if is_out_blocked:
                    answer = out_block_message
                
                st.markdown(answer)
                
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))