import streamlit as st

from chromaAPI import list_collections


def initialize_session():
    st.session_state.model = st.session_state.get(
        "model",
        "qwen2:7b"
    )
    st.session_state.collection = st.session_state.get(
        "collection",
        None
    )
    st.session_state.web_search = st.session_state.get(
        "web_search",
        False
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    
def change_value(variable, value):
    # value may be the old value
    ## NOTE: This function is used as a transition function with doing nothing
    st.session_state[variable] = value


def nav_bar(show_settings=True):
    initialize_session()
    collections = list_collections()
    if st.session_state.collection not in collections:
        if len(collections) > 0:
            if "all" in collections:
                st.session_state.collection = "all"
            else:
                st.session_state.collection = collections[0]
        else:
            st.warning("向量資料庫為空！")
            st.session_state.collection = None
    
    if "authentication_status" not in st.session_state:
        st.switch_page("rag_engine.py")
    with st.sidebar:
        if st.session_state.authentication_status:
            st.title(f" {st.session_state['name']} 您好")
            # authenticator = st.session_state.authenticator
            # authenticator.logout()
        else:
            st.title("匿名使用者")
        
        st.page_link(
            "rag_engine.py",
            label="登入介面",
            icon="🔒"
        )
        st.page_link(
            "pages/chatbot.py",
            label="對話系統",
            icon="💬"
        )
        
        if st.session_state.authentication_status:
            st.page_link(
                "pages/console.py",
                label="人工修改回饋",
                icon="📝"
            )
        
        st.page_link(
            "pages/viewer.py",
            label="資料庫內容檢視器",
            icon="📄"
        )
        
        st.page_link(
            "pages/questions.py",
            label="問題列表",
            icon="❓"
        )
        
        st.markdown(
            "[模型問答評估(外部連結)](https://docs.google.com/spreadsheets/d/1yzWKVnpBeaGXm0jSOir49OmB-O5YZxB1oSLIj1qPuug/edit?gid=640065091#gid=640065091)",
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        st.session_state.web_search = st.toggle(
            "搜尋外部資料",
            st.session_state.get("web_search", False),
            on_change=change_value,
            args=("web_search", st.session_state.web_search)
        )
        
        st.markdown("---")
        
        if show_settings:
            st.subheader("選擇模型")
            st.session_state.model = st.selectbox(
                "選擇模型",
                ["qwen2:7b", "llama3.1:8b", "llama3:latest"],
                index=[
                    "qwen2:7b", "llama3.1:8b", "llama3:latest"
                ].index(st.session_state.model),
                label_visibility="hidden",
                on_change=change_value,
                args=("model", st.session_state.model)
            )
            st.markdown("---")
            st.subheader("選擇向量資料庫")
            st.session_state.collection = st.selectbox(
                "選擇向量資料庫",
                collections,
                index=collections.index(st.session_state.collection),
                label_visibility="hidden",
                on_change=change_value,
                args=("collection", st.session_state.collection)
            )
            
            print("==={ Settings }===")
            print("model:", st.session_state.model)
            print("collection:", st.session_state.collection)
            print("web_search:", st.session_state.web_search)
            print("==================")
