from basic import nav_bar
from chromaAPI import (
    search_collection,
    create_collection
)
from databaseAPI import Feedbacks
from tools.google_search import search_google
from tools.web_retriever import search_pubmed
from tools.translator import translate

import opencc
import markdown
import pandas as pd
import streamlit as st

from pathlib import Path
from streamlit_feedback import streamlit_feedback
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser


converter = opencc.OpenCC("s2twp.json")
PARENT_DIR = Path(__file__).resolve().parent
FAQ_PATH = PARENT_DIR.joinpath("cheat.xlsx")


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


def flatten_docs(ref_docs):
    references = []
    for doc in ref_docs:
        doc_obj = {}
        for k, v in doc.metadata.items():
            if k != "source":
                doc_obj[k] = v
            else:
                doc_obj[k] = Path(v).name
        doc_obj["content"] = doc.page_content
        references.append(doc_obj)
    return references


def load_faq():
    return pd.read_excel(
        FAQ_PATH,
        engine="openpyxl",
        sheet_name="Preload_answer",
        usecols=[
            "Question",
            "Chatgpt",
            "Chatgpt_resource",
            "Perplexity",
            "Perplexity_resource",
            "Gemini",
            "Gemini_resource",
        ]
    )


def get_chatgpt(query):
    faq = load_faq()
    response = faq[faq["Question"] == query]["Chatgpt"].values[0]
    resource = faq[faq["Question"] == query]["Chatgpt_resource"].values[0]
    # convert numpy.float64 to str
    response = str(response)
    resource = str(resource)
    return response, resource


def get_perplexity(query):
    faq = load_faq()
    response = faq[faq["Question"] == query]["Perplexity"].values[0]
    resource = faq[faq["Question"] == query]["Perplexity_resource"].values[0]
    # convert numpy.float64 to str
    response = str(response)
    resource = str(resource)
    return response, resource


def get_gemini(query):
    faq = load_faq()
    response = faq[faq["Question"] == query]["Gemini"].values[0]
    resource = faq[faq["Question"] == query]["Gemini_resource"].values[0]
    # convert numpy.float64 to str
    response = str(response)
    resource = str(resource)
    return response, resource


def other_source_answer(query, stype):
    if st.session_state.gemini:
        if stype == "gemini":
            try:
                response, resource = get_gemini(query)
                return {
                    "query": query,
                    "response": response,
                    "references": [{"source": "Gemini", "content": resource}]
                }
            except Exception as e:
                print("Gemini Error:", e)
                return None
    if st.session_state.chatgpt:
        if stype == "chatgpt":
            try:
                response, resource = get_chatgpt(query)
                return {
                    "query": query,
                    "response": response,
                    "references": [{"source": "ChatGPT", "content": resource}]
                }
            except Exception as e:
                print("ChatGPT Error:", e)
                return None
    if st.session_state.perplexity:
        if stype == "perplexity":
            try:
                response, resource = get_perplexity(query)
                return {
                    "query": query,
                    "response": response,
                    "references": [{"source": "Perplexity", "content": resource}]
                }
            except Exception as e:
                print("Perplexity Error:", e)
                return None
    return None


def query_llm(query, model="qwen2:7b", temperature=0):
    model = st.session_state.model
    llm = define_llm(model, temperature)
    
    feedbacks = Feedbacks()
    rows = feedbacks.get_relevant_feedbacks(query)
    
    # if found relevant feedbacks, answer the query based on the feedbacks
    if rows:
        st.info("使用 Human Feedback 回答問題")
        ref_docs = [
            Document(
                metadata={"source": "Human Feedback"},
                page_content=str(row[0] + "\n\n" + row[1])
            )
            for row in rows
        ]
    else:
        # if user wants to search the web
        if st.session_state.web_search:
            st.info("使用 PubMed 資料回答問題")
            with st.spinner("搜尋 PubMed 資料中..."):
                en_query = translate(query)
                web_docs = search_pubmed(en_query)
                create_collection(
                    "web_retriever",
                    web_docs,
                    "english"
                )
                ref_docs = search_collection(
                    "web_retriever",
                    en_query
                )
        else:
            st.info("使用內部資料回答問題")
            with st.spinner("搜尋資料庫中的相關文件..."):
                collection = st.session_state.collection
                ref_docs = search_collection(
                    collection,
                    query
                )
    
    # generate response by ref docs
    llm_chain = add_prompt(llm)
    result = llm_chain.invoke({
        "query": query,
        "context": "\n\n".join([doc.page_content for doc in ref_docs])
    })
    if st.session_state.google_search:
        try:
            google_results = search_google(query)
            create_collection(
                "google_search",
                google_results,
            )
            google_ref_docs = search_collection(
                "google_search",
                query
            )
            llm_response_with_google = llm_chain.invoke({
                "query": query,
                "context": "\n\n".join([doc.page_content for doc in google_ref_docs])
            })
            google_answer = {
                "query": query,
                "response": converter.convert(llm_response_with_google),
                "references": flatten_docs(google_ref_docs)
            }
        except Exception as e:
            print(e)
            google_answer = None
    else:
        google_answer = None
    gemini_answer = other_source_answer(query, stype="gemini")
    chatgpt_answer = other_source_answer(query, stype="chatgpt")
    perplexity_answer = other_source_answer(query, stype="perplexity")
    model_answer = {
        "query": query,
        "response": converter.convert(result),
        "references": flatten_docs(ref_docs),
    }
    st.session_state.messages.append(
        (model_answer, google_answer, gemini_answer, chatgpt_answer, perplexity_answer)
    )
    return result


def user_feedback(feed_content, user_query, ai_response):
    good_count = 0
    bad_count = 0
    if feed_content["score"] == "👍":
        good_count = 1
    elif feed_content["score"] == "👎":
        bad_count = 1
    feedbacks = Feedbacks()
    feedbacks.insert_feedback(
        user_query, ai_response, good_count, bad_count, feed_content["text"]
    )
    st.toast("感謝您的回饋！🙏")
    
    
def display_references(references):
    expander = st.expander("查看參考文件")
    for ref in references:
        other_keys = [k for k in ref.keys() if k != "source" and k != "content" and k != "file_path"]
        content = markdown.markdown(ref["content"])
        markdown_text = f"<b>來源：</b> {ref['source']}<br>"
        for k in other_keys:
            markdown_text += f"<b>{k}：</b> {ref[k]}<br>"
        markdown_text += f"<b>內容：</b><br><pre>{content}</pre>"
        markdown_text += "<hr>"
        # print(markdown_text)
        expander.markdown(
            markdown_text,
            unsafe_allow_html=True
        )


def main():
    st.set_page_config(
        page_title="RAG QA System",
        page_icon="💬",
        layout="wide"
    )
    st.title("RAG QA System")
    nav_bar()
    
    st.chat_message("AI").write("歡迎使用 RAG QA System！")
            
    if st.session_state.collection:
        st.info("資料庫：『" + st.session_state.collection + "』使用中")
    
    # Display chat history
    for i, (
        model_answer,
        google_answer,
        gemini_answer,
        chatgpt_answer,
        perplexity_answer
    ) in enumerate(st.session_state.messages):
        st.chat_message("User").write(model_answer["query"])
        st.markdown("### LLM with RAG")
        display_references(model_answer["references"])
        st.chat_message("AI").write(model_answer["response"])
        streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="有什麼想說的嗎？",
            on_submit=user_feedback,
            key=str(i)+"_model",
            args=(model_answer["query"], model_answer["response"])
        )
        if google_answer:
            st.markdown("### LLM + Google Search")
            display_references(google_answer["references"])
            st.chat_message("AI").write(google_answer["response"])
            streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="有什麼想說的嗎？",
                on_submit=user_feedback,
                key=str(i)+"_google",
                args=(google_answer["query"], google_answer["response"])
            )
        if gemini_answer:
            st.markdown("### Gemini")
            display_references(gemini_answer["references"])
            st.chat_message("AI").write(gemini_answer["response"])
            streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="有什麼想說的嗎？",
                on_submit=user_feedback,
                key=str(i)+"_gemini",
                args=(gemini_answer["query"], gemini_answer["response"])
            )
        if chatgpt_answer:
            st.markdown("### ChatGPT")
            display_references(chatgpt_answer["references"])
            st.chat_message("AI").write(chatgpt_answer["response"])
            streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="有什麼想說的嗎？",
                on_submit=user_feedback,
                key=str(i)+"_chatgpt",
                args=(chatgpt_answer["query"], chatgpt_answer["response"])
            )
        if perplexity_answer:
            st.markdown("### Perplexity")
            display_references(perplexity_answer["references"])
            st.chat_message("AI").write(perplexity_answer["response"])
            streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="有什麼想說的嗎？",
                on_submit=user_feedback,
                key=str(i)+"_perplexity",
                args=(perplexity_answer["query"], perplexity_answer["response"])
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
