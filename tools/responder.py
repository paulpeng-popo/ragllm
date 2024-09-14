from tools.llms import define_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def rag_responder(llm=define_llm()):
    instruction = """
        Use the documents provided below to answer the user's question.
        When answer to user:
        - If you don't know, just say that you don't know.
        - If the answer to the user's question is not in the documents, just say that you don't know.
        - You are prohibited from answering questions that are not in the documents and you are not allowed to over-explain the information in the documents.
        Please answer the user's question directly and concisely.
        And answer according to the language of the user's question.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instruction),
            ("system", "documents: \n\n {documents}"),
            ("human", "question: {question}"),
        ]
    )

    return prompt | llm | StrOutputParser()


def default_responder(llm=define_llm()):
    instruction = """
        You are an assistant responsible for handling user questions. Please use your knowledge to respond to questions.
        When responding to questions, please ensure the accuracy of your answers and avoid fabricating answers.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instruction),
            ("human", "question: {question}"),
        ]
    )

    return prompt | llm | StrOutputParser()
