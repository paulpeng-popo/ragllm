import yaml # type: ignore
import streamlit as st
import streamlit_authenticator as stauth # type: ignore

from yaml.loader import SafeLoader # type: ignore
from modules.basic import nav_bar, change_page
from modules.basic import CONFIG_PATH
from streamlit_authenticator import Hasher
from streamlit_authenticator.utilities import LoginError # type: ignore


def main():
    st.set_page_config(
        page_title="登入 RAG QA 系統",
        page_icon="🔒",
        layout="wide"
    )
    st.title("亞仕丹 RAG QA 展示系統")
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=SafeLoader)

    Hasher.hash_passwords(config["credentials"])
    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        config["pre-authorized"],
    )
    
    st.session_state.model_mode = st.session_state.get("model_mode", "qwen2:7b")
    st.session_state.temperature = st.session_state.get("temperature", 0.0)
    st.session_state.query = st.session_state.get("query", "retriever")
    st.session_state.external = st.session_state.get("external", "PubMed")
    
    try:
        authenticator.login(
            fields={"Username": "使用者名稱", "Password": "密碼"},
        )
        st.session_state["authenticator"] = authenticator
    except LoginError as e:
        st.error(e)

    if st.session_state["authentication_status"]:
        change_page("chatbot")
    elif st.session_state["authentication_status"] is False:
        st.error("使用者名稱或密碼錯誤")
    elif st.session_state["authentication_status"] is None:
        if st.button("不登入直接使用"):
            change_page("chatbot")
        
    nav_bar()


if __name__ == "__main__":
    main()
