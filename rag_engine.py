import yaml
import streamlit as st
import streamlit_authenticator as stauth

from pathlib import Path
from yaml.loader import SafeLoader
from basic import nav_bar
from streamlit_authenticator import Hasher
from streamlit_authenticator.utilities import LoginError


PARENT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PARENT_DIR.joinpath("config.yaml")


def fake_login():
    st.session_state["authentication_status"] = True
    st.session_state["name"] = "測試人員"
    st.session_state["authenticator"] = None
    
    
def fake_logout():
    st.session_state["authentication_status"] = False
    st.session_state["name"] = None
    st.session_state["authenticator"] = None
    
    
def login_form():
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
    
    # try:
    #     authenticator.login(
    #         fields={
    #             "Username": "使用者名稱",
    #             "Password": "密碼"
    #         },
    #     )
    #     st.session_state.authenticator = authenticator
    # except LoginError as e:
    #     st.error(e)

    # if st.session_state.authentication_status:
    #     st.switch_page("pages/chatbot.py")
    # elif st.session_state.authentication_status is False:
    #     st.error("使用者名稱或密碼錯誤")
    #     if st.button("不登入使用"):
    #         st.switch_page("pages/chatbot.py")
    # elif st.session_state.authentication_status is None:
    #     if st.button("不登入使用"):
    #         st.switch_page("pages/chatbot.py")


def main():
    st.set_page_config(
        page_title="登入 RAG QA 系統",
        page_icon="🔒",
        layout="wide"
    )
    st.title("亞仕丹 RAG QA 展示系統")
    
    login_form()
    if st.button("我是一般使用者"):
        fake_logout()
        st.switch_page("pages/chatbot.py")
    if st.button("我是內部測試人員"):
        fake_login()
        st.switch_page("pages/chatbot.py")

    nav_bar()


if __name__ == "__main__":
    main()
