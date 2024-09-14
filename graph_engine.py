import requests

from tools.routers import question_router
from tools.graders import (
    documents_grader,
    hallucinations_grader,
    answer_grader
)
from tools.responder import rag_responder, default_responder
from tools.web_retriever import search_pubmed

from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    

### Nodes ###
def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    
    documents = retriever.invoke(question)
    return {
        "documents": documents,
        "question": question
    }


def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"] if state["documents"] else []

    web_docs = search_pubmed(question)
    documents = documents + web_docs
    return {
        "documents": documents,
        "question": question
    }


def retrieval_grade(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    documents = state["documents"]
    question = state["question"]

    filtered_docs = []
    for d in documents:
        score = documents_grader.invoke({
            "question": question,
            "document": d.page_content
        })
        grade = score.binary_score
        if grade == "yes":
            print("  -GRADE: DOCUMENT RELEVANT-")
            filtered_docs.append(d)
        else:
            print("  -GRADE: DOCUMENT NOT RELEVANT-")
            continue
    return {
        "documents": filtered_docs,
        "question": question
    }


def rag_generate(state):
    print("---GENERATE IN RAG MODE---")
    question = state["question"]
    documents = state["documents"]

    generation = rag_responder.invoke({
        "documents": documents,
        "question": question
    })
    return {
        "documents": documents,
        "question": question,
        "generation": generation
    }


def plain_answer(state):
    print("---GENERATE PLAIN ANSWER---")
    question = state["question"]
    
    generation = default_responder.invoke({
        "question": question
    })
    return {
        "question": question,
        "generation": generation
    }


### Edges ###
def route_question(state):
    print("---ROUTE QUESTION---")
    question = state["question"]
    q_router = question_router()
    source = q_router.invoke({
        "question": question
    })

    if "tool_calls" not in source.additional_kwargs:
        print("  -ROUTE TO PLAIN LLM-")
        return "plain_answer"
    if len(source.additional_kwargs["tool_calls"]) == 0:
        raise "Router could not decide source"

    datasource = source.additional_kwargs["tool_calls"][0]["function"]["name"]
    if datasource == "web_search":
        print("  -ROUTE TO WEB SEARCH-")
        return "web_search"
    elif datasource == "vectorstore":
        print("  -ROUTETO VECTORSTORE-")
        return "vectorstore"


def route_retrieval(state):
    print("---ROUTE RETRIEVAL---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        print("  -DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, ROUTE TO WEB SEARCH-")
        return "web_search"
    else:
        print("  -DECISION: GENERATE WITH RAG LLM-")
        return "rag_generate"


def grade_rag_generation(state):
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucinations_grader.invoke({
        "documents": documents,
        "generation": generation
    })
    grade = score.binary_score

    if grade == "no":
        print("  -DECISION: GENERATION IS GROUNDED IN DOCUMENTS-")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({
            "question": question,
            "generation": generation
        })
        grade = score.binary_score
        if grade == "yes":
            print("  -DECISION: GENERATION ADDRESSES QUESTION-")
            return "useful"
        else:
            print("  -DECISION: GENERATION DOES NOT ADDRESS QUESTION-")
            return "not useful"
    else:
        print("  -DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY-")
        return "not supported"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("retrieval_grade", retrieval_grade)
workflow.add_node("rag_generate", rag_generate)
workflow.add_node("plain_answer", plain_answer)

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
        "plain_answer": "plain_answer",
    },
)
workflow.add_edge("retrieve", "retrieval_grade")
workflow.add_edge("web_search", "retrieval_grade")
workflow.add_conditional_edges(
    "retrieval_grade",
    route_retrieval,
    {
        "web_search": "web_search",
        "rag_generate": "rag_generate",
    },
)
workflow.add_conditional_edges(
    "rag_generate",
    grade_rag_generation,
    {
        "not supported": "rag_generate",
        "not useful": "web_search",
        "useful": END,
    },
)
workflow.add_edge("plain_answer", END)
app = workflow.compile()


def run(question):
    inputs = {"question": question}
    for output in app.stream(inputs):
        print("\n")

    # Final generation
    if "rag_generate" in output.keys():
        print(output["rag_generate"]["generation"])
    elif "plain_answer" in output.keys():
        print(output["plain_answer"]["generation"])


if __name__ == "__main__":
    question = "台灣的首都是哪裡？"
    run(question)
