# 亞仕丹討論紀錄

## [亞士丹提供的檔案](https://drive.google.com/drive/folders/12MsN87RSMo_I4kUC9_z5tOwPtqoBCIDn?usp=sharing)
## [APP 網址](http://140.116.245.154:8501/)

## 第一階段目標
- 建立一個類似 Open web-ui 的網頁系統 (DEMO)
- 中文長文本的 RAG 能力

### 7/26 開會：實作網頁出來
- [x] 1. 監測目錄底下的 PDF 自動抓取後作 vector database (煜博 streamlit)
- [x] 2. 掃描到的文件列出
- [x] 3. Chatbot 根據 Vector database 回答 QA (煜博)
- [ ] 4. 測試 不同 embeddings, LLM (何約瑟)

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
- [x] 單一文件英文
- [x] (五個)多文件中文
- [x] (五個)多文件英文

研究 chatchat
- retreival 參數
- 他的斷詞器


### 8/13 討論

- vector store 重複建立
- ensemble retriever
- chinese splitter (猜測 改善效能有限，後續測試)
- pdf loader (rapid ocr) (後續測試)
- [hsnwlib](https://hackmd.io/@meebox/H1KsrWmh2?utm_source=preview-mode&utm_medium=rec)
- pdf 無法讀取問題
    - 圖片
    - 文字選取跳行
- demo
    - 中英 多文件 成功範例
    - 失敗的範例 ＋ 原因

### 8/16 (向亞仕丹簡報)
他們的回饋
- 讀取外部網站資料
- word 讀取, ppt 讀取
- 中文問，英文回答（翻譯）
- 回答不好，糾正 (RLHF)
    - 外部
    - 內部：編輯修正
- retriever 回傳包含：(影片、圖片、文字)
- 後端與系統商應用服務的介接


## 第二階段目標
- 輸入資料
    - 公司內部文件
        - pdf, ppt, word
        - 圖片、表格
    - 外部網站資料
        - 指定特定網址爬取內容
        - 外部資料整合進 retriever database 
- Human Feedback (RLHF) 框架
- 後續他們會希望用 line 介面介接我們的系統
    - 可能要考慮一些框架整合問題
- 預估時間：3~4個禮拜

### 分工項目
- 文件中的圖片轉文字 OCR $\Rightarrow$ (靖睿協助)
- 外部資料爬蟲整合進系統 $\Rightarrow$ (詩晴協助)
- RLHF 介面 $\Rightarrow$ (煜博、晨瑞)
    - 公司內部人員修改
        - 有過去的對話集的話，希望當成標準答案
    - 外部使用者對話回饋

---

## PDF RAG 第二階段詳細開發目標討論 (三大方向)
- RLHF
- 網頁 Retriever
- Document Loader

### RLHF

1. 確認 OpenWeb UI Session 內的 Feedback (修改、按讚、倒讚) 等流程後面的機制
   1. 對話 Session 內修改後的資料為另一個 RAG 的來源？
   2. 傳回去 Model 本身做 Fine-Tune? (較不可能)
2. 初步區分 RLHF 系統為兩大權限
   1. 公司內部檢視修改
   2. 外部使用者正常對話回饋
3. 公司內部檢視修改系統
   1. 需要帳號權限管控
   2. 對話內修改回饋需保持 Persistent 到整個系統，OpenWebUI 似乎只有在單一對話內
   3. 回饋流程是直接做 RAG 來源，還是收集大量資料後做一次 Fine-Tune? 
   4. 額外紀錄檢視介面觀看
      1. 修改過的問題列表
      2. 外部使用者傳回的回饋列表
4. 外部使用者回饋
   1. 僅開放使用者按讚或倒讚
   2. 回傳至公司內部檢視修改系統介面讓內部人員稽核檢視
5. 如果 Open-WebUI 系統只是做 RAG
   1. 如何決定要不要 Fine-Tune?
   2. Fine-Tune 的流程、方法？
   3. 如何評估效果？
   4. 如果只倚賴全系統 persistent 的 RAG 就可以解決，Fine-tune 介入的必要性？

### 網頁 Retriever

1. 系統內有可編輯界面指定要爬的網頁網址(格式暫時先存成 json 或 txt)
    - https://pubmed.ncbi.nlm.nih.gov/
    - https://www.wolterskluwer.com/en/solutions/uptodate
3. Langchain 內有不同 Retriever class, 但是與 BeautifulSoup vs Webdriver 一樣，太過簡單的 Retriever 無法爬取 JS 動態載入的 內容，有較進階的網頁 Retriever 可以處理
4. 定期自動更新機制，考慮不同方案
5. 自動更新是否有比對新舊資料機制
6. 不確定對方是要跟內部文件一起參考還是要另外獨立，建議跟文件 RAG 同一層級先行獨立實做
7. 確認中英文回答狀況

### Document Loader

1. 目標是解決（word、ppt、pdf、txt）、表格讀取、圖片內文字、英文文字跳行的問題
2. 實驗不同的 Document Loader 處理的效果，或者搜尋別人處理的經驗
3. 建議 Debug 的時候直接看處理後文字，不要經過 Model

---

### 多輪 (具有 context-aware)
- [ ] 後續功能加入討論

---

### 8/23 討論
- PDF
    1. 表格語意資訊，順序屬性
    2. 欄、列合併的特殊處理方式
    3. table loader 試其他的 loader
- PPT
    1. 圖解文字比較難讀出語意資訊
        - 可能用 **multimodal** 的模型去找
        - 有沒有辦法從圖片資訊知道，這是一張圖解圖片
        - caption
        - 語意結構
    2. OCR 中文辨識成亂碼
    3. retriever 是否有做索引
- **word/ppt 轉 pdf** vs **直接 讀 word/ppt**

## 8/29 討論
- Retriever 分類成不同種類的知識集


## 參考資料
- [Streamlit](https://github.com/langchain-ai/streamlit-agent)
- [Open webui](https://github.com/open-webui/open-webui)
- [Vector database](https://github.com/mileszim/awesome-vector-database)
- [Chinese Embeddings Compare](https://ihower.tw/blog/archives/12167/comment-page-1#comment-77856)
- [PDF-LLM-RAG](https://gitlab.ilovetogether.com/gbanyan.huang/PDF-LLM-RAG)
- [RLHF](https://huggingface.co/blog/rlhf)
- [PDF parser](https://www.reddit.com/r/LangChain/comments/1dzj5qx/best_pdf_parser_for_rag/)
- [LLaMa-Factory (RLHF)](https://qwen.readthedocs.io/en/latest/training/SFT/llama_factory.html)
    - 對話集 dataset 可以直接丟進去跑 (如果需要 finetune 模型的話)
- [RAG LangGraph](https://edge.aif.tw/application-langchain-rag-advanced/)

## 預計第三階段進度
1. 文件知識集(不同種類的器材文件建成不同叢集)
2. 加入 Mutimodal 機制
4. 系統前後端分離 / vector database API

## 接班人
- 向廷 (PM)
- 靖睿 (PM 助手)
- 王維俊 (RAG 大將)
- 何約瑟 (RAG 大將)

## 9/14 討論
- 針對一般使用者，學術文章要如何回答給普通使用者較恰當？
    - 英文文章、學術文章
    - 爬全文，還是 abstract 就好，再加上翻譯
    - 注意 entity 會不會錯: 數字、日期、...(8大類)
- 爬蟲 resources 來源
    - pubmed
    - uptodate
    - 要指定來源，還是我們系統主動爬其他相關網站？
    - 可能 pubmed 可以涵括大部分的醫療知識
- 確認亞仕丹想要的 knowledge level
    - 販賣醫療器材
    - 系統服務客戶
        - 專業問題(專業醫療人員)
        - 一般民眾()
    - 既有文件能夠準確回答
    - 相關的醫療問題 --> RLHF 修正 --> 外部準確資料庫
    - Knowledge level (三級)
        1. 內部文件
        2. 外部資料(法規)
        3. 常用、不常用的使用者問題

---

## 煜博
- Document loader 整合
- 9/16 簡報
- 使用者回饋後的問題分類

## 何約瑟、王維俊
- 知識回覆階段和制定問答範圍 (Regulate Knowledge levels)
    - 40% 60% FAQ 問題分類
    - 對象
        - 專業問答(專業醫療人員)
        - 一般民眾
        - 內部公司員工
    - 問題類型
        - 人時地物 --> 器材、健保法規(可能搭配外部檢索)
        - How
        - Why
    - 製作測試問題集
        - 生成問題＋從文件找答案
        - 每類約 5 組問題答案

## 向廷、靖睿
- 加入 Mutimodal 機制
    - llava 模型
    - 使用時機
        - 系統決定
        - 使用者決定
    - 測量花費時間、vram、記憶體、GPU usage ...等資源耗用量
