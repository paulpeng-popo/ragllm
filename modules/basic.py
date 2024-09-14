import streamlit as st

from pathlib import Path


PARENT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PARENT_DIR.joinpath("config.yaml")
DATA_DIR = PARENT_DIR.joinpath("data").joinpath("pdfdata")
DATABASE_PATH = PARENT_DIR.joinpath("data").joinpath("db.sqlite")


def nav_bar():
    if "authentication_status" not in st.session_state:
        change_page("home")
    with st.sidebar:
        if st.session_state["authentication_status"]:
            st.title(f" {st.session_state['name']} 您好")
            authenticator = st.session_state["authenticator"]
            authenticator.logout()
        else:
            st.title("匿名使用者")
        
        st.page_link("rag_engine.py", label="登入介面", icon="📄")
        st.page_link("pages/chatbot.py", label="對話系統", icon="💬")
        
        if st.session_state["authentication_status"]:
            st.page_link("pages/feedback_controller.py", label="人工修改回饋", icon="📝")
        
        st.markdown("---")
        
        st.page_link("pages/settings.py", label="設定", icon="⚙️")


def change_page(page):
    if page == "home":
        st.switch_page("rag_engine.py")
    elif page == "chatbot":
        st.switch_page("pages/chatbot.py")
    elif page == "feedback_controller":
        st.switch_page("pages/feedback_controller.py")
    elif page == "settings":
        st.switch_page("pages/settings.py")
    else:
        st.error("找不到該頁面")
