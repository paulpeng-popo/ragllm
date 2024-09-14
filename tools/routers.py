from tools.llms import define_llm
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class web_search(BaseModel):
    query: str = Field(
        description="the question to be asked to the web search tool"
    )

class vectorstore(BaseModel):
    query: str = Field(
        description="the question used to search the vector database"
    )
    

# 如果之後文件有分類，可以在 prompt 中進一步指示
def question_router(llm=define_llm()):
    instruction = """
        You are an expert in using vector database or web search to guide user questions.
        The vector database contains documents on the use of medical equipment. Use the vector database tool for questions on these topics. Otherwise, use the web search tool.
    """
    
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instruction),
            ("human", "{question}"),
        ]
    )
    
    structured_llm_router = llm.bind_tools(tools=[web_search, vectorstore])
    
    question_router = route_prompt | structured_llm_router
    return question_router
