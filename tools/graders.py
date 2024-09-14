from tools.llms import define_llm
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="whether the document is relevant to the question ('yes' or 'no')"
    )
    
    
def documents_grader(llm=define_llm()):
    instruction = """
        You are a grader responsible for evaluating the relevance of documents to user questions.
        If the document contains keywords or semantics relevant to the user question, grade it as relevant.
        Output 'yes' or 'no' to represent the relevance of the document to the question.
    """
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instruction),
            ("human", "document: \n\n {document} \n\n user question: {question}"),
        ]
    )
    
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    return grade_prompt | structured_llm_grader


class GradeHallucinations(BaseModel):
    binary_score: str = Field(
        description="whether the LLM response is hallucinated ('yes' or 'no')"
    )


def hallucinations_grader(llm=define_llm()):
    instruction = """
        You are a grader responsible for evaluating whether the LLM response is hallucinated.
        Given a document and the corresponding LLM response, output 'yes' or 'no' to indicate whether the LLM response is hallucinated.
        'Yes' indicates that the LLM response is hallucinated and not based on the content of the document. 'No' indicates that the LLM response is not hallucinated and is based on the content of the document.
    """
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instruction),
            ("human", "documents: \n\n {documents} \n\n LLM response: {generation}"),
        ]
    )
    
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    return grade_prompt | structured_llm_grader
    
    
class GradeAnswer(BaseModel):
    binary_score: str = Field(
        description="whether the LLM response answers the user's question ('yes' or 'no')"
    )


def answer_grader(llm=define_llm()):
    instruction = """
        You are a grader responsible for evaluating whether the LLM response answers the user's question.
        Given the user question and the LLM response, output 'yes' or 'no' to indicate whether the LLM response answers the user's question.
        'Yes' indicates that the LLM response answers the user's question. 'No' indicates that the LLM response does not answer the user's question.
    """
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instruction),
            ("human", "user question: {question} \n\n answer: {generation}"),
        ]
    )
    
    structured_llm_grader = llm.with_structured_output(GradeAnswer)
    return grade_prompt | structured_llm_grader
