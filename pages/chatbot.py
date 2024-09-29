from basic import nav_bar
from chromaAPI import (
    search_collection,
    create_collection
)
from databaseAPI import Feedbacks
from tools.google_search import search_google
from tools.web_retriever import search_pubmed
from tools.translator import translate

import re
import opencc
import markdown
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from pathlib import Path
from uuid import uuid4
from fuzzywuzzy import process
from load_data import get_folders, get_files_recursive
from streamlit_pdf_viewer import pdf_viewer
from streamlit_feedback import streamlit_feedback
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser


converter = opencc.OpenCC("s2twp.json")
PARENT_DIR = Path(__file__).resolve().parent
FAQ_PATH = PARENT_DIR.joinpath("cheat.xlsx")

VIDEO_RESULT = 4
IMAGE_RESULT = 4
WEB_RESULT = 2


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
        - If you don't know, just say that 『相關資料不足，提供的答案可能有所誤差及錯誤』, don't make up information.
        - Avoid over-explain the context when identifying the answer.
        - Answer the user's question directly and concisely.
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


def get_response_resource(query, stype):
    response_column_name = stype.capitalize()
    resource_column_name = stype.capitalize() + "_resource"
    faq = load_faq()
    response = faq[faq["Question"] == query][response_column_name].values[0]
    resource = faq[faq["Question"] == query][resource_column_name].values[0]
    # convert numpy.float64 to str
    response = str(response)
    resource = str(resource)
    return response, resource


def extract_links(content):
    # Extract video link from markdown format
    # [*text*](link)
    # url_extract_pattern = "https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"
    # use the shortest match
    video_links = []
    image_links = []
    others = []
    pattern = re.compile(r"\[.*?\]\((.*?)\)")
    pattern2 = "https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"
    for match in pattern.finditer(content):
        link = match.group(1)
        if link and (link.endswith(".mp4") or link.endswith(".webm") or link.endswith(".mov") or "youtube" in link):
            video_links.append(link)
        elif link and (link.endswith(".png") or link.endswith(".jpg") or link.endswith(".jpeg") or link.endswith(".gif")):
            image_links.append(link)
        else:
            others.append(link)
    print("Pattern 1:", video_links, image_links, others)
    links = re.findall(pattern2, content)
    for link in links:
        if link and (link.endswith(".mp4") or link.endswith(".webm") or link.endswith(".mov") or "youtube" in link):
            video_links.append(link)
        elif link and (link.endswith(".png") or link.endswith(".jpg") or link.endswith(".jpeg") or link.endswith(".gif")):
            image_links.append(link)
        else:
            others.append(link)
    print("Pattern 2:", video_links, image_links, others)
    # remove content with pattern
    content = re.sub(pattern, "", content)
    content = re.sub(pattern2, "", content)
    return {
        "video_links": list(set(video_links)),
        "image_links": list(set(image_links)),
        "other_links": list(set(others)),
        "content": content
    }


def other_source_answer(query, stype):
    if stype in st.session_state and st.session_state[stype]:
        try:
            response, resource = get_response_resource(query, stype)
            response_links = extract_links(response)
            resource_links = extract_links(resource)
            video_links = response_links["video_links"] + resource_links["video_links"]
            image_links = response_links["image_links"] + resource_links["image_links"]
            web_links = response_links["other_links"] + resource_links["other_links"]
            return {
                "query": query,
                "response": response_links["content"],
                "references": [{"source": stype.capitalize(), "content": resource}],
                "video_links": video_links[:VIDEO_RESULT],
                "image_links": image_links[:IMAGE_RESULT],
                "web_links": web_links[:WEB_RESULT]
            }
        except Exception as e:
            print(f"{stype.capitalize()} Error:", e)
            return None
    return None


def answer_with_company_files(model_answer):
    all_files = []
    for folder in get_folders():
        files = [
            {
                "file_path": f,
                "file_name": Path(f).name
            }
            for f in get_files_recursive(folder)
        ]
        all_files.extend(files)
    special_questions = [
        "駝人ＬＭＡ產品目錄",
        "駝人授權書",
        "拋棄式矽膠喉頭面罩目錄及仿單",
        "蘇打石灰公司授權代理資料",
        "高雄榮總蘇打石灰報價表",
        "查詢MD喉頭鏡葉片四號庫存",
        "Rota-Trach 一般氣切上課資料",
        "Rota-Trach 一般無囊氣切7.5 剩多少庫存",
        "Uniblocker支氣管內管阻隔器型錄",
        "請問Uniblocker支氣管內管阻隔器產品比較表？",
        "請問Uniblocker支氣管內管阻隔器原廠資料？",
        "請問Uniblocker支氣管內管阻隔器規格表？",
        "請問Uniblocker支氣管內管阻隔器區台大報價單？",
    ]
    query = model_answer["query"]
    question = process.extract(query, special_questions, limit=1)
    if question[0][1] > 60:
        files = [f["file_name"] for f in all_files]
        relavant_files = process.extract(query, files, limit=3)
        relavant_files = [f[0] for f in relavant_files]
        relavant_files = list(set(relavant_files))
        # get these file path
        relavant_files_path = [
            f["file_path"]
            for f in all_files
            if f["file_name"] in relavant_files
        ]
        model_answer["relavant_files"] = relavant_files_path
    
    return model_answer


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
    model_answer = {
        "query": query,
        "response": converter.convert(result),
        "references": flatten_docs(ref_docs),
    }
    model_answer = answer_with_company_files(model_answer)
    
    # search web data
    if st.session_state.pubmed_search:
        st.info("外部搜尋：搜尋 PubMed 資料")
        with st.spinner("搜尋 PubMed 資料中..."):
            en_query = translate(query)
            web_docs = search_pubmed(en_query)
            create_collection(
                "web_retriever",
                web_docs,
                "english"
            )
            pubmed_ref_docs = search_collection(
                "web_retriever",
                en_query
            )
    else:
        pubmed_ref_docs = []
    # search google data
    if st.session_state.google_search:
        st.info("外部搜尋：搜尋 Google 資料")
        try:
            web_docs = search_google(query)
            create_collection(
                "google_search",
                web_docs,
            )
            google_ref_docs = search_collection(
                "google_search",
                query
            )
        except Exception as e:
            print(e)
            google_ref_docs = []
    else:
        google_ref_docs = []
        
    # combine web search references
    if st.session_state.pubmed_search or st.session_state.google_search:
        web_search_ref_docs = pubmed_ref_docs + google_ref_docs
        llm_response = llm_chain.invoke({
            "query": query,
            "context": "\n\n".join([
                doc.page_content for doc in ref_docs + web_search_ref_docs
            ])
        })
        web_search_answer = {
            "query": query,
            "response": converter.convert(llm_response),
            "references": flatten_docs(ref_docs + web_search_ref_docs),
            "web_links": [
                doc.metadata["link"]
                for doc in web_search_ref_docs
            ][:WEB_RESULT]
        }
    else:
        web_search_answer = None
    gemini_answer = other_source_answer(query, stype="gemini")
    chatgpt_answer = other_source_answer(query, stype="chatgpt")
    perplexity_answer = other_source_answer(query, stype="perplexity")
    
    st.session_state.messages.append(
        (
            model_answer,
            web_search_answer,
            gemini_answer,
            chatgpt_answer,
            perplexity_answer
        )
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
    with st.expander("查看參考文件"):
        # if all([k in ref.keys() for ref in references for k in ["source", "file_path", "page"]]):
        if False:
            col1, col2 = st.columns(2)
            already_displayed = []
            for i, ref in enumerate(references):
                if ref["file_path"] in already_displayed:
                    continue
                if i % 2 == 0:
                    col1.markdown(
                        f"<b>來源：</b> {ref['source']}<br>"
                        f"<b>檔案：</b> {ref['file_path']}",
                        unsafe_allow_html=True
                    )
                    pdf_viewer(ref["file_path"], pages_to_render=[ref["page"]])
                    col1.markdown("<hr>", unsafe_allow_html=True)
                else:
                    col2.markdown(
                        f"<b>來源：</b> {ref['source']}<br>"
                        f"<b>檔案：</b> {ref['file_path']}",
                        unsafe_allow_html=True
                    )
                    pdf_viewer(ref["file_path"], pages_to_render=[ref["page"]])
                    col2.markdown("<hr>", unsafe_allow_html=True)
                already_displayed.append(ref["file_path"])
            return
        else:
            for i, ref in enumerate(references):
                # other_keys = [k for k in ref.keys() if k != "source" and k != "content" and k != "file_path"]
                content = markdown.markdown(ref["content"])
                markdown_text = f"<b>來源：</b> {ref['source']}<br>"
                # for k in other_keys:
                #     markdown_text += f"<b>{k}：</b> {ref[k]}<br>"
                markdown_text += f"<b>內容：</b><br><pre>{content}</pre>"
                markdown_text += "<hr>"
                # print(markdown_text)
                # st.markdown(
                #     markdown_text,
                #     unsafe_allow_html=True
                # )
                col1, col2 = st.columns(2)
                if i % 2 == 0:
                    col1.markdown(markdown_text, unsafe_allow_html=True)
                else:
                    col2.markdown(markdown_text, unsafe_allow_html=True)
            return


def display_links(links, link_type, height=150):
    if len(links) == 0:
        return
    link_not_show = [
        "pubs.asahq.org",
        "ncbi.nlm.nih.gov",
    ]
    st.markdown(f"#### {link_type}")
    col1, col2 = st.columns(2)
    for i, link in enumerate(links):
        if i % 2 == 0:
            with col1:
                if link_type == "video":
                    st.video(link)
                elif link_type == "image":
                    st.image(link)
                else:
                    st.markdown(link)
                    if not any([k in link for k in link_not_show]):
                        components.iframe(link, height=height)
        else:
            with col2:
                if link_type == "video":
                    st.video(link)
                elif link_type == "image":
                    st.image(link)
                else:
                    st.markdown(link)
                    if not any([k in link for k in link_not_show]):
                        components.iframe(link, height=height)


def show_pdf_files(file_paths):
    for file_path in file_paths:
        if not Path(file_path).as_posix().endswith(".pdf"):
            continue
        filename = Path(file_path).name
        with open(file_path, "rb") as f:
            btn = st.download_button(
                label=f"📄 {filename}",
                data=f,
                file_name=filename,
                key=uuid4(),
                mime="application/pdf"
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
        st.info("資料庫：『 " + st.session_state.collection + " 』使用中")
    
    # Display chat history
    for i, (
        model_answer,
        web_search_answer,
        gemini_answer,
        chatgpt_answer,
        perplexity_answer
    ) in enumerate(st.session_state.messages):
        st.chat_message("User").write(model_answer["query"])
        if web_search_answer:
            main_col_1, main_col_2 = st.columns(2)
            with main_col_1:
                with st.container(border=True):
                    st.markdown("### 公司內部文件搜尋")
                    display_references(model_answer["references"])
                    st.chat_message("AI").write(model_answer["response"])
                    show_pdf_files(model_answer.get("relavant_files", []))
                    streamlit_feedback(
                        feedback_type="thumbs",
                        optional_text_label="有什麼想說的嗎？",
                        on_submit=user_feedback,
                        key=str(i)+"_model",
                        args=(model_answer["query"], model_answer["response"])
                    )
            with main_col_2:
                with st.container(border=True):
                    st.markdown("### Google + PubMed 搜尋")
                    display_references(web_search_answer["references"])
                    st.chat_message("AI").write(web_search_answer["response"])
                    streamlit_feedback(
                        feedback_type="thumbs",
                        optional_text_label="有什麼想說的嗎？",
                        on_submit=user_feedback,
                        key=str(i)+"_web",
                        args=(web_search_answer["query"], web_search_answer["response"])
                    )
                    display_links(web_search_answer["web_links"], "web", height=300)
        else:
            with st.container(border=True):
                st.markdown("### 公司內部文件搜尋")
                display_references(model_answer["references"])
                st.chat_message("AI").write(model_answer["response"])
                show_pdf_files(model_answer.get("relavant_files", []))
                streamlit_feedback(
                    feedback_type="thumbs",
                    optional_text_label="有什麼想說的嗎？",
                    on_submit=user_feedback,
                    key=str(i),
                    args=(model_answer["query"], model_answer["response"])
                )
        main_col_1, main_col_2, main_col_3 = st.columns(3)
        if gemini_answer:
            with main_col_1:
                with st.container(border=True):
                    st.markdown("### 詢問 Gemini 的結果")
                    # display_references(gemini_answer["references"])
                    st.chat_message("AI").write(gemini_answer["response"])
                    streamlit_feedback(
                        feedback_type="thumbs",
                        optional_text_label="有什麼想說的嗎？",
                        on_submit=user_feedback,
                        key=str(i)+"_gemini",
                        args=(gemini_answer["query"], gemini_answer["response"])
                    )
                    display_links(gemini_answer["video_links"], "video")
                    display_links(gemini_answer["image_links"], "image")
                    display_links(gemini_answer["web_links"], "web")
        if chatgpt_answer:
            with main_col_2:
                with st.container(border=True):
                    st.markdown("### 詢問 ChatGPT 的結果")
                    # display_references(chatgpt_answer["references"])
                    st.chat_message("AI").write(chatgpt_answer["response"])
                    streamlit_feedback(
                        feedback_type="thumbs",
                        optional_text_label="有什麼想說的嗎？",
                        on_submit=user_feedback,
                        key=str(i)+"_chatgpt",
                        args=(chatgpt_answer["query"], chatgpt_answer["response"])
                    )
                    display_links(chatgpt_answer["video_links"], "video")
                    display_links(chatgpt_answer["image_links"], "image")
                    display_links(chatgpt_answer["web_links"], "web")
        if perplexity_answer:
            with main_col_3:
                with st.container(border=True):
                    st.markdown("### 詢問 Perplexity 的結果")
                    # display_references(perplexity_answer["references"])
                    st.chat_message("AI").write(perplexity_answer["response"])
                    streamlit_feedback(
                        feedback_type="thumbs",
                        optional_text_label="有什麼想說的嗎？",
                        on_submit=user_feedback,
                        key=str(i)+"_perplexity",
                        args=(perplexity_answer["query"], perplexity_answer["response"])
                    )
                    display_links(perplexity_answer["video_links"], "video")
                    display_links(perplexity_answer["image_links"], "image")
                    display_links(perplexity_answer["web_links"], "web")
    
    # Chat input
    if query := st.chat_input():
        st.chat_message("User").write(query)
        with st.spinner("AI 正在思考中..."):
            response = query_llm(query)
        st.chat_message("AI").write(response)
        st.rerun()


if __name__ == "__main__":
    main()
