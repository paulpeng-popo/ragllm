import os
from pathlib import Path

import pdfplumber
import ppt2pdf
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader

from pypdf import PdfReader
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageStat
import io, os
from fpdf import FPDF

from rag_engine import extract_tables_from_pdf, split_documents, create_ensemble_retriever
def is_image_black(image):
    # 检查图像是否是全黑
    stat = ImageStat.Stat(image)
    if sum(stat.extrema[0]) == 0:
        return True
    return False


DATA_DIR = Path(__file__).resolve().parent.joinpath(".")
VECTOR_STORE_DIR = DATA_DIR.joinpath("vector_store")
PDF_DATA_DIR = DATA_DIR.joinpath("pptdata")


def extract_tables_from_pdf(pdf_path, page_number):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        # print("page_number:", pdf.pages[page_number])
        tables.extend(pdf.pages[page_number].extract_tables())
    return tables


def extract_images_and_text_from_pdf(pdf_path, page_number):
    pdf_name = os.path.basename(pdf_path)
    pdf_name = os.path.splitext(pdf_name)[0]
    # 打开 PDF 文件
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number)
    images_with_text = []
    # 提取页面中的图片
    image_list = page.get_images(full=True)
    for img_index, img in enumerate(image_list):
        xref = img[0]
        base_image = pdf_document.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]

        # 将图像加载到 PIL 图像对象
        image = Image.open(io.BytesIO(image_bytes))

        # 检查是否为全黑图像
        if is_image_black(image):
            continue  # 跳过全黑图像
        # 将图片存储为文件
        image_filename = f"{pdf_name}_image_{page_number + 1}_{img_index + 1}.{image_ext}"
        with open(image_filename, "wb") as image_file:
            image_file.write(image_bytes)

        # 打开图像文件进行 OCR
        # image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)

        # 将图片和对应的文字一起存储
        images_with_text.append({'image_filename': image_filename, 'text': text})

    # with PdfReader(pdf_path) as pdf:
    #     page = pdf.pages[page_number]
    #     for i in page.images:
    #         with open(i.name, "wb") as f:
    #             f.write(i.data)

    return images_with_text


def process_documents():
    file_loader = PyPDFDirectoryLoader(
        path=PDF_DATA_DIR.as_posix(),
        glob="**/*.pdf"
    )
    try:
        with st.spinner("製作向量資料庫..."):
            documents = file_loader.load()
            all_doc = []
            for doc in documents:
                pdf_path = dict(doc)["metadata"]["source"]
                page_num = dict(doc)["metadata"]["page"]
                # 提取PDF中的表格数据
                tables = extract_tables_from_pdf(pdf_path, page_num)

                # 提取PDF中的图片和文字
                images_with_text = extract_images_and_text_from_pdf(pdf_path, page_num)

                # 将表格和图片（含文字）数据附加到文档元数据中
                doc.metadata['tables'] = tables
                doc.metadata['images_with_text'] = images_with_text

                # print("doc", doc.metadata)
                all_doc.append(doc)
            print("all_doc:", all_doc)
            docs = split_documents(documents)
            # st.session_state.retriever = create_retriever(docs)
            # st.session_state.retriever = parent_doc_retrevier(docs)
            st.session_state.retriever = create_ensemble_retriever(docs)
        st.success("資料庫製作完成")
        print("資料庫製作完成")
    except Exception as e:
        print(e)
        st.error(f"處理文件時出現錯誤: {e}")


from pptxtopdf import convert

if __name__ == "__main__":
    # 指定PPT文件路径和输出PDF文件路径
    ppt_file = "./pptdata/影像化插管.pptx"
    pdf_file = "./pptdata/"
    # print("success")
    # 执行转换
    convert(ppt_file, pdf_file)
    process_documents()
