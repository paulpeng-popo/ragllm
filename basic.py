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
    st.session_state.pubmed_search = st.session_state.get(
        "pubmed_search",
        False
    )
    st.session_state.google_search = st.session_state.get(
        "google_search",
        False
    )
    st.session_state.gemini = st.session_state.get(
        "gemini",
        False
    )
    st.session_state.chatgpt = st.session_state.get(
        "chatgpt",
        False
    )
    st.session_state.perplexity = st.session_state.get(
        "perplexity",
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
            # st.title("匿名使用者")
            pass
        
        st.page_link(
            "rag_engine.py",
            label="登入",
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
            
        if show_settings:
            st.session_state.model = st.selectbox(
                # icon="🔧",
                "🔧 選擇模型",
                ["qwen2:7b", "llama3.1:8b", "llama3:latest"],
                index=[
                    "qwen2:7b", "llama3.1:8b", "llama3:latest"
                ].index(st.session_state.model),
                on_change=change_value,
                args=("model", st.session_state.model)
            )
            st.session_state.collection = st.selectbox(
                # icon="🔧",
                "🔧 選擇向量資料庫",
                collections,
                index=collections.index(st.session_state.collection),
                on_change=change_value,
                args=("collection", st.session_state.collection)
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
        
        # st.markdown(
        #     "[模型問答評估(外部連結)](https://docs.google.com/spreadsheets/d/1yzWKVnpBeaGXm0jSOir49OmB-O5YZxB1oSLIj1qPuug/edit?gid=640065091#gid=640065091)",
        #     unsafe_allow_html=True
        # )
        
        # st.markdown(
        #     "[問題答案(外部連結)](https://docs.google.com/spreadsheets/d/1sU6effzDWO_3JAlm1nGvkqG3o8QyTbLr/edit?gid=1305424167#gid=1305424167)",
        #     unsafe_allow_html=True
        # )
        
        st.subheader("外部資源列表")
        
        st.session_state.pubmed_search = st.toggle(
            "搜尋 PubMed",
            st.session_state.get("pubmed_search", False),
            on_change=change_value,
            args=("pubmed_search", st.session_state.pubmed_search)
        )
        
        st.session_state.google_search = st.toggle(
            "搜尋 Google",
            st.session_state.get("google_search", False),
            on_change=change_value,
            args=("google_search", st.session_state.google_search)
        )
        
        st.session_state.gemini = st.toggle(
            "詢問 Gemini",
            st.session_state.get("gemini", False),
            on_change=change_value,
            args=("gemini", st.session_state.gemini)
        )
        
        st.session_state.chatgpt = st.toggle(
            "詢問 ChatGPT",
            st.session_state.get("chatgpt", False),
            on_change=change_value,
            args=("chatgpt", st.session_state.chatgpt)
        )
        
        st.session_state.perplexity = st.toggle(
            "詢問 Perplexity",
            st.session_state.get("perplexity", False),
            on_change=change_value,
            args=("perplexity", st.session_state.perplexity)
        )
            
        print("==={ Settings }===")
        print("model:", st.session_state.model)
        print("collection:", st.session_state.collection)
        print("pubmed_search:", st.session_state.pubmed_search)
        print("google_search:", st.session_state.google_search)
        print("gemini:", st.session_state.gemini)
        print("chatgpt:", st.session_state.chatgpt)
        print("perplexity:", st.session_state.perplexity)
        print("==================")
