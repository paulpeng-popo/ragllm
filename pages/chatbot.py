from modules.dbtool import DBTool
from modules.basic import nav_bar
from modules.basic import DATABASE_PATH
from modules.chinese_splitter import ChineseRecursiveTextSplitter
from tools.web_retriever import search_pubmed
from tools.translator import translate

import requests
import opencc # type: ignore
import pandas as pd # type: ignore

converter = opencc.OpenCC("s2twp.json")
dbtool = DBTool(db_name=DATABASE_PATH.as_posix())

import streamlit as st

from pathlib import Path
from streamlit_feedback import streamlit_feedback # type: ignore
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser


def define_llm(mode="qwen2:7b", temperature=0):
    if mode == "qwen2:7b":
        return ChatOllama(model="qwen2:7b", temperature=temperature)
    elif mode == "llama3.1:8b":
        return ChatOllama(model="llama3.1:8b", temperature=temperature)
    elif mode == "llama3:8b":
        return ChatOllama(model="llama3:8b", temperature=temperature)


def add_prompt(llm):
    RAG_TEMPLATE = """
        Use the following context as your learned knowledge, inside <context></context> XML tags.
        <context>
            {context}
        </context>

        When answer to user:
        - If you don't know, just say that you don't know.
        - If you don't know when you are not sure, ask for clarification.
        Avoid over-explain the context when answering the user's question.
        Please answer the user's question directly and concisely.
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


def query_llm(query, model_mode="qwen2:7b", temperature=0):
    model_mode = st.session_state.model_mode
    temperature = st.session_state.temperature
    print(model_mode, temperature)
    llm = define_llm(model_mode, temperature)
    retriever_database = get_human_retriever(query)
    if st.session_state.web_search:
        with st.spinner("搜尋外部資料中..."):
            en_query = translate(query)
            web_docs = search_pubmed(en_query)
            ref_docs = split_documents(web_docs)
            ref_docs = [
                doc.dict()
                for doc in ref_docs
            ]
    else:
        with st.spinner("AI 搜尋資料庫中..."):
            ref_docs = requests.get(
                "http://140.116.245.154:8510/" + retriever_database + "/" + query,
            ).json()
    llm_chain = add_prompt(llm)
    result = llm_chain.invoke({
        "query": query,
        "context": "\n\n".join([doc["page_content"] for doc in ref_docs])
    })
    references = [
        {
            "filename": Path(doc["metadata"]["source"]).name,
            "page": doc["metadata"]["page"],
            "content": doc["page_content"],
        }
        for doc in ref_docs
    ]
    expander = st.expander("查看參考文件")
    for ref in references:
        expander.info(f"{ref['filename']} 第 {ref['page']} 頁\n\n{ref['content']}")
    result = converter.convert(result)
    st.session_state.messages.append((query, result, references))
    return result


def split_documents(documents):
    # Recursivesplitter = RecursiveCharacterTextSplitter(
    #     separators=["\n\n", "\n", "。"],
    #     chunk_size=1500,
    #     chunk_overlap=100,
    #     keep_separator=False,
    # )
    # docs = Recursivesplitter.split_documents(documents)
    ChineseSplitter = ChineseRecursiveTextSplitter(
        is_separator_regex=True,
        chunk_size=1500,
        chunk_overlap=100,
        keep_separator=False,
    )
    docs = ChineseSplitter.split_documents(documents)
    return docs


def user_feedback(feed_content, user_query, ai_response):
    good_count = 0
    bad_count = 0
    if feed_content["score"] == "👍":
        good_count = 1
    elif feed_content["score"] == "👎":
        bad_count = 1
    dbtool.insert(
        "feedbacks",
        [user_query, ai_response, good_count, bad_count, feed_content["text"]]
    )
    st.toast("感謝您的回饋！🙏")
    
    
def get_human_retriever(question):
    data = dbtool.select_data("feedbacks")
    if len(data) == 0 or question not in [row[1] for row in data]:
        st.session_state.query = "retriever"
        return st.session_state.query
    rows_we_need = [row for row in data if row[1] == question]
    res = requests.post(
        "http://140.116.245.154:8510/human",
        json={
            "documents": [
                {
                    "source": "human_feedback_answer",
                    "page": row[0],
                    "page_content": str(row[1] + "\n\n" + row[2] + "\n\n" + row[5])
                }
                for row in rows_we_need
            ]
        }
    )
    print(res.json())
    st.session_state.query = "human"
    return st.session_state.query


def main():
    st.set_page_config(
        page_title="RAG QA System",
        page_icon="💬",
        layout="wide"
    )
    st.title("RAG QA System")
    nav_bar()
    
    st.session_state.model_mode = st.session_state.get("model_mode", "qwen2:7b")
    st.session_state.temperature = st.session_state.get("temperature", 0.0)
    st.session_state.query = st.session_state.get("query", "retriever")
    st.session_state.external = st.session_state.get("external", "PubMed")
    
    # Initialize session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.web_search = st.toggle("搜尋外部資料" + " (目前: " + st.session_state.external + ")", False)
    st.chat_message("AI").write("歡迎使用 RAG QA System！")
    
    # Display chat history
    for i, (user_msg, ai_msg, references) in enumerate(st.session_state.messages):
        st.chat_message("User").write(user_msg)
        expander = st.expander("查看參考文件")
        for ref in references:
            expander.info(f"{ref['filename']} 第 {ref['page']} 頁\n\n{ref['content']}")
        st.chat_message("AI").write(ai_msg)
        streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="有什麼想說的嗎？",
            on_submit=user_feedback,
            key=str(i),
            args=(user_msg, ai_msg)
        )
    
    # Chat input
    if query := st.chat_input():
        st.chat_message("User").write(query)
        with st.spinner("AI 正在思考中..."):
            response = query_llm(query)
        st.chat_message("AI").write(response)
        st.rerun()


if __name__ == "__main__":
    main()
