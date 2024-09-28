from pathlib import Path
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader


PARENT_DIR = Path(__file__).resolve().parent
DATA_DIR = PARENT_DIR.joinpath("data")


def get_folders():
    # Get all folders in the data directory
    folders = [folder for folder in DATA_DIR.iterdir() if folder.is_dir()]
    yield from folders
    
    
def get_files_recursive(folder):
    # Get all files in the folder and its subfolders
    for file in folder.iterdir():
        if file.is_file():
            yield file
        elif file.is_dir():
            yield from get_files_recursive(file)
            
            
def classify_files(file_paths):
    pic_types = [".jpg", ".jpeg", ".png"]
    video_types = [".mp4"]
    pdf_types = [".pdf"]
    ppt_types = [".ppt", ".pptx"]
    doc_types = [".doc", ".docx"]
    xls_types = [".xls", ".xlsx"]
    
    pic_files = []
    video_files = []
    pdfs = []
    ppts = []
    words = []
    excels = []
    others = []
    
    for file_path in file_paths:
        file_ext = file_path.suffix.lower()
        if file_ext in pic_types:
            pic_files.append(file_path)
        elif file_ext in video_types:
            video_files.append(file_path)
        elif file_ext in pdf_types:
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
        "ppts": ppts,
        "words": words,
        "excels": excels,
        "others": others #ignore
    }


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


def process_files(file_types):
    pdfs = file_types["pdfs"]
    pdf_documents = []
    for pdf_path in pdfs:
        pdf_loader = PDFLoader(str(pdf_path))
        pdf_documents.extend(pdf_loader.load())
        
    ppts = file_types["ppts"]
    ppt_documents = []
    for ppt_path in ppts:
        ppt_loader = PowerPointLoader(str(ppt_path))
        ppt_documents.extend(ppt_loader.load())
        
    words = file_types["words"]
    word_documents = []
    for word_path in words:
        word_loader = WordLoader(str(word_path))
        word_documents.extend(word_loader.load())
        
    excels = file_types["excels"]
    excel_documents = []
    for excel_path in excels:
        excel_loader = ExcelLoader(str(excel_path))
        excel_documents.extend(excel_loader.load())
        
    return {
        "pdfs": pdf_documents,
        "ppts": ppt_documents,
        "words": word_documents,
        "excels": excel_documents,
    }


if __name__ == "__main__":
    from chromaAPI import create_collection
    for i, folder in enumerate(get_folders()):
        print(f"Processing folder {i+1}: {folder.name}")
        files = [f for f in get_files_recursive(folder)]
        file_types = classify_files(files)
        document_types = process_files(file_types)
        collection_name = folder.name
        for doc_type, documents in document_types.items():
            print(f"Processing {len(documents)} {doc_type} documents...")
            create_collection(collection_name, documents)
    
    # all folder create a collection named "all"
    all_documents = []
    for folder in get_folders():
        files = [f for f in get_files_recursive(folder)]
        file_types = classify_files(files)
        document_types = process_files(file_types)
        for doc_type, documents in document_types.items():
            all_documents.extend(documents)
    create_collection("all", all_documents)
    
    print("All documents processed.")
