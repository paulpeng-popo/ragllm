# Description: Streamlit app for RAG QA system

# Import built-in libraries
import tempfile
from pathlib import Path

import warnings

# Filter specific warning messages
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set", category=FutureWarning)
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`", category=FutureWarning)

# Import third-party libraries
import jieba
import hanlp
import pandas as pd
from chinese_converter import to_traditional

electra_base = hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH
tokenizer = hanlp.load(electra_base, devices=0)

# Import Streamlit
import streamlit as st

# Import Langchain modules
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.storage import InMemoryStore
from langchain.prompts import PromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_text_splitters.base import TextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama


# File paths
DATA_DIR = Path(__file__).resolve().parent.joinpath("data")
VECTOR_STORE_DIR = DATA_DIR.joinpath("vector_store")
PDF_DATA_DIR = DATA_DIR.joinpath("pdfdata")

# Project setup
st.set_page_config(
    page_title="PDF RAG QA 系統",
    page_icon="📚",
    layout="wide"
)
st.title("亞仕丹 RAG QA 展示系統")

mode = st.sidebar.selectbox(
    "使用模型",
    ("qwen2:7b", "llama3.1:8b", "llama3:8b", "openAI"),
)
st.sidebar.info(f"更換至 {mode} 模型")

def define_llm():
    if mode == "qwen2:7b":
        return ChatOllama(model="qwen2:7b")
    elif mode == "llama3.1:8b":
        return ChatOllama(model="llama3.1:8b")
    elif mode == "llama3:8b":
        return ChatOllama(model="llama3:8b")
    elif mode == "openAI":
        return ChatOpenAI(model="gpt-4o")

def add_prompt(llm):
    ENGLISH_TEMPLATE = """
        you are helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision. \
        Provide an answer to the following question completely based on provide data. \
        Retrieve the answer only in the provided infomartion. \
        Do not rely on the knowledge in your original trained data, \
        Ensure that the answer is, \
        relevant, and concise, \
        output the answer on Chinese \
        {query}
    """
    CHINESE_TEMPLATE = """
        你是一個樂於助人、善良、誠實、擅長寫作、並且從不失敗地立即且精確地回答任何請求的人。 \
        請完全根據提供的資訊回答以下問題。 \
        並只在提供的資訊中搜尋答案。 \
        不要依賴於你原始訓練數據中的知識， \
        確保答案是，相關且簡潔的，輸出答案為中文 \
        {query}
    """
    RAG_TEMPLATE = """
        Use the following context as your learned knowledge, inside <context></context> XML tags.
        <context>
            {context}
        </context>

        When answer to user:
        - If you don't know, just say that you don't know.
        - If you don't know when you are not sure, ask for clarification.
        Avoid mentioning that you obtained the information from the context.
        And answer according to the language of the user's question.

        Given the context information, answer the query.
        Query: {query}
    """
    
    prompt_template = RAG_TEMPLATE
    input_prompt = PromptTemplate(
        input_variables=["query", "context"],
        template=prompt_template
    )

    return input_prompt | llm | StrOutputParser()

def query_llm(retriever, query):
    llm = define_llm()
    ref_docs = retriever.invoke(query)
    llm_chain = add_prompt(llm)
    result = llm_chain.invoke({
        "query": query,
        "context": "\n\n".join([doc.page_content for doc in ref_docs])
    })
    references = [
        {
            "filename": Path(doc.metadata["source"]).name,
            "page": doc.metadata["page"],
            "content": doc.page_content,
        }
        for doc in ref_docs
    ]
    expander = st.expander("查看參考文件")
    for ref in references:
        expander.info(f"{ref['filename']} 第 {ref['page']} 頁\n\n{ref['content']}")
    st.session_state.messages.append((query, result))
    return result

def show_created_files():
    file_list = []
    with open("file_list.txt", "r") as f:
        for line in f:
            file_list.append(line.strip())
    # show file list with dataframe
    df = pd.DataFrame(
        file_list,
        columns=["向量資料庫內容"],
        index=range(1, len(file_list) + 1)
    )
    st.sidebar.dataframe(df)
        
class JiebaTextSplitter(TextSplitter):
    def __init__(self, separator: str = "\n\n", **kwargs):
        super().__init__(**kwargs)
        self._separator = separator

    def split_text(self, text: str):
        splits = jieba.lcut(text)
        return self._merge_splits(splits, self._separator)
    
class HanLPTextSplitter(TextSplitter):
    def __init__(self, separator: str = "\n\n", **kwargs):
        super().__init__(**kwargs)
        self._separator = separator

    def split_text(self, text: str):
        doc = tokenizer(text, tasks=["tok/fine"])
        splits = doc["tok/fine"]
        return self._merge_splits(splits, self._separator)
    
def split_documents(documents):
    Recursivesplitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。"],
        chunk_size=1500,
        chunk_overlap=100,
        keep_separator=False,
    )
    JiebaSplitter = JiebaTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
    )
    HanLPSplitter = HanLPTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
    )
    docs = Recursivesplitter.split_documents(documents)
    return docs

def create_retriever(docs):
    model_name = "BAAI/bge-m3"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"}
    )
    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR.as_posix(),
        collection_name="pdf_retriever"
    )
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 7, 'lambda_mult': 0.25}
    )
    ## https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.chroma.Chroma.html
    return retriever

def parent_doc_retrevier(docs):
    model_name = "BAAI/bge-m3"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"}
    )
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR.as_posix(),
        collection_name="pdf_retriever"
    )
    child_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。"],
        chunk_size=1500,
        chunk_overlap=100,
        keep_separator=False,
    )
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )
    retriever.add_documents(docs)
    # sub_docs = vectorstore.similarity_search("駝人影像式插管組包含哪些零件？")
    # for sub_doc in sub_docs:
    #     print(sub_doc.page_content)
    #     print("="*50)
    # retrieved_docs = retriever.invoke("駝人影像式插管組包含哪些零件？")
    # for doc in retrieved_docs:
    #     print(doc.page_content)
    return retriever

def process_documents():
    file_loader = PyPDFDirectoryLoader(
        path=PDF_DATA_DIR.as_posix(),
        glob="**/*.pdf"
    )
    try:
        with st.status("製作向量資料庫...", expanded=True) as status:
            st.write("讀取文件...")
            documents = file_loader.load()
            st.write("分割資料...")
            docs = split_documents(documents)
            st.write("建立 Retriever...")
            # st.session_state.retriever = create_retriever(docs)
            st.session_state.retriever = parent_doc_retrevier(docs)
            status.update(
                label="向量資料庫製作完成",
                state="complete",
                expanded=False
            )
        file_list = [doc.metadata["source"] for doc in documents]
        file_list = list(set(file_list))
        file_list = [Path(vecfile).name for vecfile in file_list]
        # save to file
        with open("file_list.txt", "w") as f:
            for item in file_list:
                f.write("%s\n" % item)
    except Exception as e:
        st.error(f"處理文件時出現錯誤: {e}")

def initialize_session(uploader=False):
    if "messages" not in st.session_state:
        st.session_state.messages = []
        if not uploader:
            process_documents()
            st.success("Retriever 已建立")
    show_created_files()
    
    if uploader:
        # 上傳文件建立 Retriever
        st.session_state.source_docs = st.sidebar.file_uploader(
            label="選擇模型參考資料",
            type="pdf",
            accept_multiple_files=True
        )
        if not st.session_state.source_docs:
            st.warning("未建立 Retriever")
        for source_doc in st.session_state.source_docs:
            # 寫入暫存資料夾
            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=PDF_DATA_DIR.as_posix(),
                suffix=".pdf"
            ) as tmp_file:
                tmp_file.write(source_doc.read())
            # 處理文件
            process_documents()
            # 刪除暫存資料
            for _file in PDF_DATA_DIR.iterdir():
                temp_file = PDF_DATA_DIR.joinpath(_file)
                temp_file.unlink()
        st.success("Retriever 已建立")

def display_chat_history():
    for user_msg, ai_msg in st.session_state.messages:
        st.chat_message("User").write(user_msg)
        st.chat_message("AI").write(ai_msg)

def main():
    initialize_session()
    display_chat_history()
    if query := st.chat_input():
        st.chat_message("User").write(query)
        if "retriever" in st.session_state:
            st.info("使用 retriever 查詢答案")
            response = query_llm(st.session_state.retriever, query)
        else:
            st.warning("未建立 Retriever")
            response = "未建立 Retriever"
        st.chat_message("AI").write(to_traditional(response))

if __name__ == "__main__":
    main()
