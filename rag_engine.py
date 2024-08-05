import os
from pathlib import Path

import streamlit as st
from opencc import OpenCC  # type: ignore

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama

# Constants
cc = OpenCC("s2twp.json")
TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath("data", "vector_store")
PDF_DATA_DIR = Path(__file__).resolve().parent.joinpath("data", "pdfdata")

# Create necessary directories
if not TMP_DIR.exists():
    TMP_DIR.mkdir(parents=True)

# Streamlit setup
st.set_page_config(page_title="PDF RAG QA 系統", page_icon="📚", layout="wide")
st.title("亞仕丹 RAG QA 展示系統")

# Sidebar for model selection
mode = st.sidebar.radio("模型選擇", ("Llama", "openAI"))

def load_documents():
    loader = PyPDFDirectoryLoader(path=TMP_DIR.as_posix(), glob="**/*.pdf")
    return loader.load()

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

def create_embeddings(texts):
    model_name = "BAAI/bge-m3"
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
    vectordb = Chroma.from_documents(texts, embedding=embeddings, persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    return vectordb.as_retriever(search_kwargs={"k": 7})

def define_llm():
    if mode == "Llama":
        return ChatOllama(model="llama3")
    elif mode == "openAI":
        return ChatOpenAI(model="gpt-4o")

def get_prompt(query):
    # prompt_template = """
    # 你是一位專業的醫療器材專家，你的回答皆是基於給予的文件資訊，並且確保回答是正確的。 
    # 請使用繁體中文回答以下問題，請只要回答問題就好。確保答案具有意義、相關性和簡潔性：
    # """
    prompt_template = """
    你是一個能夠立即且準確地回答任何請求的人。請用中文回答以下問題。確保答案內容符合提供的資料且簡要正確： \n
    """
    return prompt_template + query

def query_llm(retriever, query):
    prompt = get_prompt(query)
    llm = define_llm()
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)
    result = qa_chain.invoke({"question": prompt, "chat_history": st.session_state.messages})["answer"]
    st.session_state.messages.append((query, result))
    return result

def process_documents():
    try:
        if any(PDF_DATA_DIR.glob("**/*.pdf")):
            loader = PyPDFDirectoryLoader(path=PDF_DATA_DIR.as_posix(), glob="**/*.pdf")
            documents = loader.load()
            texts = split_documents(documents)
            st.session_state.retriever = create_embeddings(texts)
            st.success("資料庫已更新")
        else:
            st.info("沒有新的 PDF 文件")
    except Exception as e:
        st.error(f"處理文件時出現錯誤: {e}")

def initialize_session():
    if "messages" not in st.session_state:
        process_documents()
        st.session_state.messages = []

def display_chat_history():
    for user_msg, ai_msg in st.session_state.messages:
        st.chat_message("User").write(user_msg)
        st.chat_message("AI").write(ai_msg)

def main():
    initialize_session()
    display_chat_history()
    if query := st.chat_input():
        st.chat_message("User").write(query)
        if "retriever" in st.session_state: # 要確定有跑進來
            response = query_llm(st.session_state.retriever, query)
        else:
            response = "資料庫尚未建立"
        st.chat_message("AI").write(cc.convert(response))

if __name__ == "__main__":
    main()
