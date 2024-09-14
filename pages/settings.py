## pub-med
### https://pubmed.ncbi.nlm.nih.gov/
## uptodate
### https://www.wolterskluwer.com/en/solutions/uptodate

from modules.dbtool import DBTool
from modules.basic import nav_bar
from modules.basic import DATABASE_PATH

import requests
import pandas as pd # type: ignore

dbtool = DBTool(db_name=DATABASE_PATH.as_posix())

import streamlit as st


def main():
    st.set_page_config(
        page_title="Settings",
        page_icon="🔧",
        layout="wide"
    )
    st.title("系統設定")
    nav_bar()
        
    doc_names = requests.get(
        "http://140.116.245.154:8510/retriever",
    ).json()
    
    collections = requests.get(
        "http://140.116.245.154:8510"
    ).json()["collections"]

    df = pd.DataFrame(
        doc_names,
        columns=["文件名稱"],
        index=range(1, len(doc_names) + 1)
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("設定模型")
        st.session_state.model_mode = st.selectbox(
            "選擇模型",
            ["qwen2:7b", "llama3.1:8b", "llama3:8b"],
            index=["qwen2:7b", "llama3.1:8b", "llama3:8b"].index(st.session_state.model_mode),
            label_visibility="hidden"
        )
        st.markdown("---")
        st.subheader("選擇溫度")
        st.session_state.temperature = st.slider(
            "選擇溫度 (model_temperature)",
            0.0, 1.0,
            value=st.session_state.temperature,
            step=0.1,
            label_visibility="hidden"
        )
        st.markdown("---")
        st.subheader("選擇向量資料庫")
        st.session_state.query = st.selectbox(
            "選擇向量資料庫",
            collections,
            index=collections.index(st.session_state.query),
            label_visibility="hidden"
        )
        st.markdown("---")
        st.subheader("外部資料庫預設網址")
        st.session_state.external = st.radio(
            "選擇外部資料庫",
            ["PubMed", "UpToDate"],
            index=["PubMed", "UpToDate"].index(st.session_state.external),
            label_visibility="hidden"
        )
    
    with col2:
        st.subheader("向量資料庫內容")
        st.dataframe(df, use_container_width=True, height=500)


if __name__ == "__main__":
    main()
