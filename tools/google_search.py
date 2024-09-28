import os
import nest_asyncio

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
nest_asyncio.apply()

from googlesearch import search
from langchain_core.documents import Document
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.document_loaders import WebBaseLoader


def search_google(
    query,
    num_results=5,
    sleep_interval=1,
    advanced=True
):
    results = search(
        query,
        lang="zh-TW",
        num_results=num_results,
        sleep_interval=sleep_interval,
        advanced=advanced,
        timeout=10
    )
    urls = [result.url for result in results]
    print(f"{len(urls)} sources found.")
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
    term = "洗腎導管操作影片"
    docs = search_google(term)
    for doc in docs:
        print(doc.metadata)
        print(doc.page_content[:100])
        print()
