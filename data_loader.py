import os
import io
import fitz
import pymupdf
import camelot
import requests
import traceback
import pdfplumber
import pytesseract

from pathlib import Path
from PIL import Image, ImageStat
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader


PARENT_DIR = Path(__file__).resolve().parent
DATA_DIR = PARENT_DIR.joinpath("data")


def get_folders():
    # Get all folders in the data directory
    folders = [folder for folder in DATA_DIR.iterdir() if folder.is_dir()]
    return folders
    
    
def get_files_recursive(folder):
    # Get all files in the folder and its subfolders
    for file in folder.iterdir():
        if file.is_file():
            yield file
        elif file.is_dir():
            yield from get_files_recursive(file)
            
            
def classify_files(file_paths):
    pic_types = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
    video_types = [".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv"]
    ppt_types = [".ppt", ".pptx"]
    doc_types = [".doc", ".docx"]
    xls_types = [".xls", ".xlsx"]
    pic_files = []
    video_files = []
    pdfs = []
    excels = []
    words = []
    ppts = []
    others = []
    for file_path in file_paths:
        file_ext = file_path.suffix.lower()
        if file_ext in pic_types:
            pic_files.append(file_path)
        elif file_ext in video_types:
            video_files.append(file_path)
        elif file_ext == ".pdf":
            pdfs.append(file_path)
        elif file_ext in xls_types:
            excels.append(file_path)
        elif file_ext in doc_types:
            words.append(file_path)
        elif file_ext in ppt_types:
            ppts.append(file_path)
        else:
            others.append(file_path)
    return {
        "pictures": pic_files, # ocr, multi-modal
        "videos": video_files, #ignore
        "pdfs": pdfs,
        "excels": excels,
        "words": words,
        "ppts": ppts,
        "others": others #ignore
    }
    
    
class Image2Text:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = Image.open(image_path)
        
    def extract_text(self):
        return pytesseract.image_to_string(self.image)
    

class TableExtractor:
    def __init__(self, pdf_path, page_number):
        self.pdf_path = pdf_path
        self.page_number = page_number
        self.pdf_document = fitz.open(pdf_path)
        self.page = self.pdf_document.load_page(page_number)
        
    def extract_tables(self):
        tables = camelot.read_pdf(self.pdf_path, pages=str(self.page_number + 1))
        # to markdown format
        return tables[0].df.to_markdown()


class PDFLoader:
    def __init__(self, path):
        self.path = path
        self.loader = PDFPlumberLoader(path)
        
    def load(self):
        return self.loader.load()
    
    
class ExcelLoader:
    def __init__(self, path):
        self.path = path
        self.loader = UnstructuredExcelLoader(path)
        
    def load(self):
        return self.loader.load()
    
    
class WordLoader:
    def __init__(self, path):
        self.path = path
        self.loader = UnstructuredWordDocumentLoader(path)
        
    def load(self):
        return self.loader.load()
    
    
class PowerPointLoader:
    def __init__(self, path):
        self.path = path
        self.loader = UnstructuredPowerPointLoader(path)
        
    def load(self):
        return self.loader.load()


def is_image_black(image):
    stat = ImageStat.Stat(image)
    if sum(stat.extrema[0]) == 0:
        return True
    return False


def extract_images_and_text_from_pdf(pdf_path, page_number):
    pdf_name = os.path.basename(pdf_path)
    pdf_name = os.path.splitext(pdf_name)[0]
    
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number)
    images_with_text = []
    
    image_list = page.get_images(full=True)
    for img_index, img in enumerate(image_list):
        xref = img[0]
        base_image = pdf_document.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image = Image.open(io.BytesIO(image_bytes))
        if is_image_black(image):
            continue
        text = pytesseract.image_to_string(image)
        images_with_text.append({"text": text})

    return images_with_text


def process_files(file_types):
    pdfs = file_types["pdfs"]
    pdf_documents = []
    for pdf_path in pdfs:
        pdf_loader = PDFLoader(str(pdf_path))
        pdf_documents.extend(pdf_loader.load())
        
    excels = file_types["excels"]
    excel_documents = []
    for excel_path in excels:
        excel_loader = ExcelLoader(str(excel_path))
        excel_documents.extend(excel_loader.load())
        
    words = file_types["words"]
    word_documents = []
    for word_path in words:
        word_loader = WordLoader(str(word_path))
        word_documents.extend(word_loader.load())
        
    ppts = file_types["ppts"]
    ppt_documents = []
    for ppt_path in ppts:
        ppt_loader = PowerPointLoader(str(ppt_path))
        ppt_documents.extend(ppt_loader.load())
        
    return {
        "pdfs": pdf_documents,
        # "excels": excel_documents,
        # "words": word_documents,
        # "ppts": ppt_documents
    }


    # for doc in documents:
    #     pdf_path = doc.metadata["source"]
    #     page_num = doc.metadata["page"]
    #     tables = extract_tables_from_pdf(pdf_path, page_num)
    #     images_with_text = extract_images_and_text_from_pdf(pdf_path, page_num)
    #     doc.metadata["tables"] = tables
    #     doc.metadata["images_with_text"] = images_with_text


if __name__ == "__main__":
    folders = get_folders()
    for folder in folders:
        files = [file for file in get_files_recursive(folder)]
        file_types = classify_files(files)
        # file_types = classify_files(file_paths)
        # documents = process_files(file_types)
        # print(documents)
