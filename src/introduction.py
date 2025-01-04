"""Introduction."""

from langchain_ollama import ChatOllama

from src.settings import llm_model_url, llm_model_name, llm_model_temperature

chat: ChatOllama = ChatOllama(
    base_url=llm_model_url,
    model=llm_model_name,
    temperature=llm_model_temperature
)

message: str = 'Hello, world!'
