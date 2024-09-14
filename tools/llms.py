from langchain_community.chat_models import ChatOllama


def define_llm(mode="qwen2:7b", temperature=0):
    if mode == "qwen2:7b":
        return ChatOllama(model="qwen2:7b", temperature=temperature)
    elif mode == "llama3.1:8b":
        return ChatOllama(model="llama3.1:8b", temperature=temperature)
    elif mode == "llama3:8b":
        return ChatOllama(model="llama3:8b", temperature=temperature)
