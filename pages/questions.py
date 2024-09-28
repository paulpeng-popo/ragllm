from basic import nav_bar
from pathlib import Path

import json
import streamlit as st


PARENT_DIR = Path(__file__).resolve().parent
QUESTION_PATH = PARENT_DIR.joinpath("questions.json")


def main():
    st.set_page_config(
        page_title="Question List",
        page_icon="❓",
        layout="wide"
    )
    st.title("使用之問題列表")
    nav_bar()
    
    with open(QUESTION_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    st.json(
        questions,
        expanded=True
    )


if __name__ == "__main__":
    main()
