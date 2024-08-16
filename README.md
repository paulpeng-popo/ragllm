# 亞仕丹

## 目標
- 建立一個類似 Open web-ui 的網頁系統 (DEMO)
- 中文長文本的 RAG 能力
- [測試檔案 pdf](https://drive.google.com/drive/folders/12MsN87RSMo_I4kUC9_z5tOwPtqoBCIDn?usp=sharing)

### 7/26 開會：實作網頁出來

- [x] 1. 監測目錄底下的 PDF 自動抓取後作 vector database (煜博 streamlit)
- [x] 2. 掃描到的文件列出
- [x] 3. Chatbot 根據 Vector database 回答 QA (煜博)
- [ ] 4. 測試 不同 embeddings, LLM, (何約瑟)

參考資料：

- [Streamlit](https://github.com/langchain-ai/streamlit-agent)
- [Open webui](https://github.com/open-webui/open-webui)
- [Vector database](https://github.com/mileszim/awesome-vector-database)
- [Chinese Embeddings Compare](https://ihower.tw/blog/archives/12167/comment-page-1#comment-77856)
- [PDF-LLM-RAG](https://gitlab.ilovetogether.com/gbanyan.huang/PDF-LLM-RAG)
- [RLHF](https://huggingface.co/blog/rlhf)
- [PDF parser](https://www.reddit.com/r/LangChain/comments/1dzj5qx/best_pdf_parser_for_rag/)

### 未正式討論

- RLHF 流程
- GraphRAG


### 7/31 進度報告

- [ ] 規劃開發任務目標、技術（需要研究的部分）、分工及完成時間
- [x] 1. 檢查簡繁轉換
    - 結果：學長測試 opencc 有可能部分簡體字無法轉換成繁體
    - 使用 chinese-converter 套件 (未驗證效果)
- [x] 2. Debug 沒抓到答案的原因
- [ ] 3. 測試 text splitter
- [x] 4. PDF 後面的內容會不會影響穩定度 (找到原因)
- [ ] 5. 測其他的 中文 model + embedding
- [x] 6. 找學弟測英文 prompt 效果

### 8/8 改善
- model (qwen2:7b)和 embedding 不變
- 主要影響 retriever 和 text splitter
    - top k 搜尋演算法可能只搜尋到前面的 pdf page
    - 確認 splitter 是否包含我們需要的關鍵字
    - jieba / hanlp 斷詞 + langchain recursive splitter
    - unstructured loader (後續)
- [chatchat](https://github.com/chatchat-space/Langchain-Chatchat/tree/master)
- [as_retriever](https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.chroma.Chroma.html#langchain_community.vectorstores.chroma.Chroma.as_retriever)

### 8/10 討論

DEMO 重點 (這兩天)
- [x] 單一文件中文
- [ ] 單一文件英文
- [x] (五個)多文件中文
- [ ] (五個)多文件英文

研究 chatchat
- retreival 參數
- 他的斷詞器


8/13
- vector store 重複建立
- ensemble retriever
- chinese splitter (猜測 改善效能有限，後續測試)
- pdf loader (rapid ocr pdf loader) (後續測試)
- [hsnwlib](https://hackmd.io/@meebox/H1KsrWmh2?utm_source=preview-mode&utm_medium=rec)
- pdf 無法讀取問題
    - 圖片
    - 文字選取跳行
- demo
    - 中英 多文件 成功範例
    - 失敗的範例 ＋ 原因

demo 過後，第二階段目標
1. 


## RLHF 參考資料
* https://qwen.readthedocs.io/en/latest/training/SFT/example.html
    * 對話集 dataset 可以直接丟進去跑
* LLaMa-Factory
