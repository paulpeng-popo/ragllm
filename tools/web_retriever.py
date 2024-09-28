import os
import requests
import nest_asyncio

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
nest_asyncio.apply()

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.document_loaders import WebBaseLoader


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}


def search_pubmed(term, top_k=3):
    url = "https://pubmed.ncbi.nlm.nih.gov"
    response = requests.get(url, headers=headers, params={
        "term": term,
        "filter": "simsearch2.ffrft",
    })
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("a", class_="docsum-title")
    article_metadatas = []
    for article in articles:
        response = requests.get(url + article["href"], headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        try:
            link = soup.find("div", class_="full-text-links-list").find_all("a", class_="link-item")[-1]["href"]
        except:
            continue
        article_metadata = {
            "title": article.text.strip(),
            "url": link,
        }
        article_metadatas.append(article_metadata)
    urls = [
        article_metadata["url"]
        for article_metadata in article_metadatas
    ][:top_k]
    loader = WebBaseLoader(
        urls,
        default_parser="html.parser",
        requests_per_second=10,
    )
    docs = loader.aload()
    docs = [
        Document(
            page_content=str(doc.page_content).replace("\n\n", " ").strip(),
            metadata={
                "source": str(doc.metadata["title"]).strip(),
                "link": str(doc.metadata["source"]).strip(),
            }
        )
        for doc in docs
    ]
    return docs


if __name__ == "__main__":
    term = "Root Canal Treatment"
    docs = search_pubmed(term)
    for doc in docs:
        print(doc.metadata)
        print(doc.page_content[:100])
        print()
