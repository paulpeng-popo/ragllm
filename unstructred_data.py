from langchain_unstructured import UnstructuredLoader
import nltk
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.doc import partition_docx
from unstructured.partition.text import partition_text
from unstructured.partition.pptx import partition_pptx
import shutil
import os

# 重新下载 punkt 数据包
# nltk.download('punkt')
file_paths = [
    "./pdfdata/金泰克簡易操作小卡-喉頭鏡.pdf",
    "./pptdata/影像化插管.pptx",
    "./txtdata/questions.txt",
    "./worddata/專利申請1.docx"
]

pdf_elements = partition_pdf(file_paths[0], strategy="hi_res", lang=["eng", "chi_tra"])
pptx_elements = partition_pptx(file_paths[1], strategy="hi_res")
txt_elements = partition_text(file_paths[2], strategy="hi_res")
word_elements = partition_docx(file_paths[3], strategy="hi_res")


for element in pdf_elements:
    print(f"{element.category.upper()}: {element.text}")

print("-"*50)

for element in pptx_elements:
    print(f"{element.category.upper()}: {element.text}")
print("-"*50)
for element in txt_elements:
    print(f"{element.category.upper()}: {element.text}")
print("-"*50)
for element in word_elements:
    print(f"{element.category.upper()}: {element.text}")


import camelot


def get_tables(path: str, pages):
    for page in pages:
        table_list = camelot.read_pdf(path, pages=str(page))
        if table_list.n > 0:
            for tab in range(table_list.n):
                # Conversion of the the tables into the dataframes.
                table_df = table_list[tab].df

                table_df = (
                    table_df.rename(columns=table_df.iloc[0])
                    .drop(table_df.index[0])
                    .reset_index(drop=True)
                )
                print(type(table_df))
                table_df = table_df.apply(lambda x: x.str.replace('\n', ''))
                # print(table_df)
                # Change column names to be valid as XML tags
                table_df.columns = [col.replace('\n', ' ').replace(' ', '') for col in table_df.columns]
                print(table_df.columns)
                table_df.columns = [col.replace('(', '').replace(')', '') for col in table_df.columns]
                print(table_df.columns)
                return table_df
# loader = UnstructuredLoader(file_paths)
# docs = loader.load()
# print(docs)
# tables = [el for el in elements if el.category == "Table"]
# images = [el for el in elements if el.category == "Image"]

# df = get_tables(file_paths[0], pages=[5])
# print(df)
# from IPython.display import display
# from IPython.display import Image
# display(tables[0].to_dict())
# Image(filename="cropped_images/table-3-1.jpg", width=600)
