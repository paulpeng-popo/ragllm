import asyncio
import chromadb

from uuid import uuid4
from hashlib import sha256
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Dict, Union

from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from modules.chinese_splitter import ChineseRecursiveTextSplitter


EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"}
)
METADATA = {"hnsw:space": "cosine"}
        

app = FastAPI()


main_app_lifespan = app.router.lifespan_context
@asynccontextmanager
async def lifespan_wrapper(app):
    print("sub startup")
    async with main_app_lifespan(app) as maybe_state:
        yield maybe_state
    print("sub shutdown")
app.router.lifespan_context = lifespan_wrapper


chroma_client = chromadb.HttpClient()


# 可能要有中文和英文的分割器
def split_documents(documents):
    # Recursivesplitter = RecursiveCharacterTextSplitter(
    #     separators=["\n\n", "\n", "。"],
    #     chunk_size=1500,
    #     chunk_overlap=100,
    #     keep_separator=False,
    # )
    # docs = Recursivesplitter.split_documents(documents)
    ChineseSplitter = ChineseRecursiveTextSplitter(
        is_separator_regex=True,
        chunk_size=1500,
        chunk_overlap=100,
        keep_separator=False,
    )
    docs = ChineseSplitter.split_documents(documents)
    return docs


class AddDocumentsRequest(BaseModel):
    documents: List[Dict[str, Union[str, int]]]


@app.get("/", response_class=JSONResponse)
def list_collections():
    collections = chroma_client.list_collections()
    collection_names = [collection.name for collection in collections]
    return {"collections": collection_names}


@app.get("/reset", response_class=JSONResponse)
def reset():
    try:
        chroma_client.reset()
    except Exception as e:
        return JSONResponse(
            status_code=500, content={
                "message": "Reset failed"
            }
        )
    return {
        "message": "Reset successful"
    }
    

@app.get("/{collection_name}", response_class=JSONResponse)
def get_collection(collection_name: str):
    vectorstore = Chroma(
        client=chroma_client,
        embedding_function=EMBEDDINGS,
        collection_name=collection_name,
        collection_metadata=METADATA
    )
    doc_names = [
        Path(meta_info["source"]).name 
        for meta_info in vectorstore.get()["metadatas"]
    ]
    doc_names = list(set(doc_names))
    return doc_names


@app.post("/{collection_name}", response_class=JSONResponse)
def create_collection(collection_name: str, data: AddDocumentsRequest):
    vectorstore = Chroma(
        client=chroma_client,
        embedding_function=EMBEDDINGS,
        collection_name=collection_name,
        collection_metadata=METADATA
    )
    data_list = data.documents
    documents = [
        Document(
            metadata={
                "source": doc["source"],
                "page": doc["page"],
            },
            page_content=str(doc["page_content"]),
        )
        for doc in data_list
    ]
    docs = split_documents(documents)
    unique_ids = [
        sha256(
            f"{doc.metadata['source']}_{doc.metadata['page']}_{doc.page_content}".encode()
        ).hexdigest()
        for doc in docs
    ]
    vectorstore.add_documents(documents=docs, ids=unique_ids)
    return {
        "status": "success",
        "message": f"Added {len(documents)} documents to collection {collection_name}"
    }


@app.get("/{collection_name}/{query}", response_class=JSONResponse)
def search_collection(collection_name: str, query: str):
    vectorstore = Chroma(
        client=chroma_client,
        embedding_function=EMBEDDINGS,
        collection_name=collection_name,
        collection_metadata=METADATA
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.4, 'k': 4}
    )
    results = retriever.invoke(query)
    return results


@app.delete("/{collection_name}", response_class=JSONResponse)
def delete_collection(collection_name: str):
    try:
        chroma_client.delete_collection(collection_name)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={
                "message": f"Collection {collection_name} does not exist"
            }
        )
    return {
        "message": f"Dropped collection {collection_name}"
    }
