"""Introduction."""

from langchain_ollama import ChatOllama

from src.settings import model, temperature

chat: ChatOllama = ChatOllama(model=model, temperature=temperature)
message: str = 'Hello, world!'
