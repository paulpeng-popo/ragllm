import chromadb

from pathlib import Path
from hashlib import sha256, md5
from typing import List

from databaseAPI import CollectionNames, DocumentIDs
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chinese_splitter import ChineseRecursiveTextSplitter


METADATA = {"hnsw:space": "cosine"}
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"}
)


def split_documents(documents, mode="chinese"):
    if mode == "chinese":
        Splitter = ChineseRecursiveTextSplitter(
            is_separator_regex=True,
            chunk_size=1000,
            chunk_overlap=100,
            keep_separator=False,
        )
    else:
        Splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            keep_separator=False,
        )
    docs = Splitter.split_documents(documents)
    return docs


def list_collections(mode="name"):
    chroma_client = chromadb.HttpClient()
    collections = chroma_client.list_collections()
    name_hashes = [c.name for c in collections]
    collection_names = CollectionNames()
    if mode == "name":
        names = [
            collection_names.get_name_by_hash(name_hash)
            for name_hash in name_hashes
        ]
        names = sorted(names)
        return names
    name_hashes = sorted(name_hashes)
    return name_hashes
    

def get_collection(collection_name: str):
    chroma_client = chromadb.HttpClient()
    collection_names = CollectionNames()
    hash_name = collection_names.get_hash_by_name(collection_name)
    vectorstore = Chroma(
        client=chroma_client,
        embedding_function=EMBEDDINGS,
        collection_name=hash_name,
        collection_metadata=METADATA
    )
    doc_names = [
        Path(meta_info["source"]).name 
        for meta_info in vectorstore.get()["metadatas"]
    ]
    doc_names = list(set(doc_names))
    return doc_names


def create_collection(
    collection_name: str,
    documents: List[Document],
    document_mode="chinese",
):
    chroma_client = chromadb.HttpClient()
    hash_str = md5(collection_name.encode()).hexdigest()
    collection_names = CollectionNames()
    collection_names.insert_collection_name(hash_str, collection_name)
    vectorstore = Chroma(
        client=chroma_client,
        embedding_function=EMBEDDINGS,
        collection_name=hash_str,
        collection_metadata=METADATA
    )
    docs = split_documents(documents, mode=document_mode)
    source_ids_map = {}
    docs_to_add = []
    unique_ids = []
    for doc in docs:
        source = Path(doc.metadata["source"]).name
        metadata_str = " ".join(f"{k}={v}" for k, v in doc.metadata.items())
        # str_to_hash = f"{metadata_str} {doc.page_content}"
        str_to_hash = f"{doc.page_content}"
        docid = sha256(str_to_hash.encode()).hexdigest()
        if docid in unique_ids:
            continue
        unique_ids.append(docid)
        docs_to_add.append(doc)
        if source not in source_ids_map:
            source_ids_map[source] = []
        source_ids_map[source].append(docid)
    
    if len(docs_to_add) == 0 or len(unique_ids) == 0:
        return
    
    vectorstore.add_documents(
        documents=docs_to_add,
        ids=unique_ids
    )
    
    document_ids = DocumentIDs()
    for source, ids in source_ids_map.items():
        for docid in ids:
            document_ids.insert_document_id(docid, source)


def search_collection(
    collection_name: str,
    query: str,
):
    chroma_client = chromadb.HttpClient()
    collection_names = CollectionNames()
    hash_name = collection_names.get_hash_by_name(collection_name)
    vectorstore = Chroma(
        client=chroma_client,
        embedding_function=EMBEDDINGS,
        collection_name=hash_name,
        collection_metadata=METADATA
    )
    retriever = vectorstore.as_retriever(
        # search_type="mmr",
        # search_kwargs={"k": 4, "lambda_mult": 0.25}
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.4, 'k': 4}
    )
    results = retriever.invoke(query)
    return results


def delete_collection(hash_name: str):
    try:
        chroma_client = chromadb.HttpClient()
        chroma_client.delete_collection(hash_name)
    except Exception as e:
        print(e)
        
        
def deleteAll():
    collections = list_collections("hash")
    for collection in collections:
        delete_collection(collection)


if __name__ == "__main__":
    deleteAll()
    collections = list_collections()
    print(collections)
