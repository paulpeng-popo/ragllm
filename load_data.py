import requests
import traceback

from modules.basic import DATA_DIR
from langchain_community.document_loaders import PyPDFDirectoryLoader


# def extract_tables_from_pdf(pdf_path, page_number):
#     tables = []
#     with pdfplumber.open(pdf_path) as pdf:
#         # print("page_number:", pdf.pages[page_number])
#         tables.extend(pdf.pages[page_number].extract_tables())
#     return tables


# # import fitz
# import pytesseract
# from PIL import Image, ImageStat
# import io, os


# def is_image_black(image):
#     stat = ImageStat.Stat(image)
#     if sum(stat.extrema[0]) == 0:
#         return True
#     return False

# def extract_images_and_text_from_pdf(pdf_path, page_number):
#     pdf_name = os.path.basename(pdf_path)
#     pdf_name = os.path.splitext(pdf_name)[0]
    
#     pdf_document = fitz.open(pdf_path)
#     page = pdf_document.load_page(page_number)
#     images_with_text = []
    
#     image_list = page.get_images(full=True)
#     for img_index, img in enumerate(image_list):
#         xref = img[0]
#         base_image = pdf_document.extract_image(xref)
#         image_bytes = base_image["image"]
#         image_ext = base_image["ext"]
#         image = Image.open(io.BytesIO(image_bytes))

#         if is_image_black(image):
#             continue
        
#         image_filename = f"{pdf_name}_image_{page_number + 1}_{img_index + 1}.{image_ext}"
#         with open(image_filename, "wb") as image_file:
#             image_file.write(image_bytes)

#         text = pytesseract.image_to_string(image)
#         images_with_text.append({'image_filename': image_filename, 'text': text})

#     return images_with_text


def process_documents(collection_name="retriever"):
    file_loader = PyPDFDirectoryLoader(
        path=DATA_DIR.as_posix(),
        glob="**/*.pdf"
    )
    try:
        documents = file_loader.load()
        res = requests.post(
            "http://140.116.245.154:8510/" + collection_name,
            json={
                "documents": [
                    {
                        "source": doc.metadata["source"],
                        "page": doc.metadata["page"],
                        "page_content": doc.page_content
                    }
                    for doc in documents
                ]
            }
        )
        print(res.json())
        # for doc in documents:
        #     pdf_path = doc.metadata["source"]
        #     page_num = doc.metadata["page"]
        #     tables = extract_tables_from_pdf(pdf_path, page_num)
        #     images_with_text = extract_images_and_text_from_pdf(pdf_path, page_num)
        #     doc.metadata["tables"] = tables
        #     doc.metadata["images_with_text"] = images_with_text
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        
        
def remove_collection(collection_name):
    res = requests.delete(f"http://140.116.245.154:8510/{collection_name}")
    print(res.json())


if __name__ == "__main__":
    process_documents()
    # remove_collection("retriever")
