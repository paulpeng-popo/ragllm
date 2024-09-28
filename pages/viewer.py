from basic import nav_bar
from chromaAPI import get_collection

import pandas as pd
import streamlit as st
    

def get_image_by_doc_type(doc_type: str):
    if doc_type in ["pdf"]:
        image = "app/static/pdf.png"
    elif doc_type in ["doc", "docx"]:
        image = "app/static/docx.png"
    elif doc_type in ["ppt", "pptx"]:
        image = "app/static/pptx.png"
    elif doc_type in ["xls", "xlsx"]:
        image = "app/static/xlsx.png"
    else:
        image = "app/static/unknown.png"
    return image


def get_column_config():
    return {
        "document_id": st.column_config.TextColumn(
            "文件編號",
            max_chars=5,
            width="small",
        ),
        "document_name": st.column_config.TextColumn(
            "文件名稱",
            max_chars=10,
            width="medium",
        ),
        "document_type": st.column_config.ImageColumn(
            "文件類型",
            width="small",
        ),
    }


def main():
    st.set_page_config(
        page_title="Database Viewer",
        page_icon="📄",
        layout="wide"
    )
    st.title("資料庫內容檢視器")
    nav_bar()
    
    if st.session_state.collection is not None:
        collection = st.session_state.collection
        doc_names = get_collection(collection)
        doc_amount = len(doc_names)
        df = pd.DataFrame(
            {
                "document_id": range(1, len(doc_names) + 1),
                "document_name": doc_names,
                "document_type": [
                    get_image_by_doc_type(x.split(".")[-1])
                    for x in doc_names
                ]
            }
        )
        st.subheader(f"{collection} 文件列表 ({doc_amount} 筆資料)")
        st.dataframe(
            df,
            column_config=get_column_config(),
            use_container_width=True,
            hide_index=True,
            height=400
        )
    else:
        st.warning("未選擇資料庫！")


if __name__ == "__main__":
    main()
