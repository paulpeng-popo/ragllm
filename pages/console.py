import pandas as pd
import streamlit as st

from databaseAPI import Feedbacks
from basic import nav_bar


def get_data():
    feedbacks = Feedbacks()
    rows = feedbacks.get_all_feedbacks()
    feedbacks.close()
    return pd.DataFrame(
        [
            {
                "rowid": i + 1,
                "question": row[0],
                "answer": row[1],
                "feel_correct": row[2],
                "feel_incorrect": row[3],
                "feedback_text": row[4],
            }
            for i, row in enumerate(rows)
        ],
        columns=[
            "rowid",
            "question",
            "answer",
            "feel_correct",
            "feel_incorrect",
            "feedback_text",
        ]
    )


def get_column_config():
    return {
        "rowid": st.column_config.TextColumn(
            "ID",
            max_chars=3,
            width="small",
        ),
        "question": st.column_config.TextColumn(
            "使用者問題",
            max_chars=10,
            width="medium",
        ),
        "answer": st.column_config.TextColumn(
            "模型回答",
            max_chars=20,
            width="medium",
        ),
        "feel_correct": st.column_config.NumberColumn(
            "使用者喜歡",
            format="%d 人",
            width="small",
        ),
        "feel_incorrect": st.column_config.NumberColumn(
            "使用者不喜歡",
            format="%d 人",
            width="small",
        ),
        "feedback_text": st.column_config.TextColumn(
            "訊息內容",
            max_chars=10,
            width="small",
        ),
    }
    
    
def save_button(output):
    old_data = st.session_state.selected_row
    st.session_state.pop("selected_row")
    st.session_state.changed_data = {
        "question": old_data["question"],
        "response": output,
    }


def main():
    st.set_page_config(
        page_title="Human Feedback 介面",
        page_icon="📊",
        layout="wide"
    )
    st.title("Human Feedback 介面")
    nav_bar()
    
    if "changed_data" in st.session_state:
        changed_data = st.session_state.changed_data
        with st.spinner("更新回覆中..."):
            feedbacks = Feedbacks()
            status = feedbacks.update_feedback(
                changed_data["question"],
                changed_data["response"],
            )
            feedbacks.close()
            if status:
                st.toast("更新成功！🎉")
            else:
                st.toast("更新失敗！😢")
        st.session_state.pop("changed_data")
        st.rerun()
    
    if "selected_row" not in st.session_state or \
        st.session_state.selected_row is None:
            
        st.subheader("使用者模型回覆回饋數")
        st.warning("勾選一列來編輯模型回覆")
        
        data = get_data()
        event = st.dataframe(
            data,
            column_config=get_column_config(),
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        if event.selection.rows:
            selected_row = data.iloc[event.selection.rows[0]]
            st.session_state.selected_row = selected_row
            st.rerun()
    else:
        st.subheader("編輯人工回覆")
        selected_row = st.session_state.selected_row
        st.chat_message("User").write(
            selected_row["question"]
        )
        with st.chat_message("AI"):
            output = st.text_area(
                "模型回覆",
                value=selected_row["answer"],
                height=500,
                label_visibility="hidden",
            )
        st.button(
            "儲存並返回",
            on_click=save_button,
            args=(output,)
        )


if __name__ == "__main__":
    main()
