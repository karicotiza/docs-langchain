"""Build a simple LLM application with chat models and prompt templates
module."""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.settings import model, temperature

chat: ChatOllama = ChatOllama(model=model, temperature=temperature)

messages: list[HumanMessage | SystemMessage] = [
    SystemMessage('Translate the following from English into Italian'),
    HumanMessage('hi!'),
]

string_message: str = 'Hello'
dict_messages: list[dict[str, str]] = [{'role': 'user', 'content': 'Hello'}]
human_messages: list[HumanMessage] = [HumanMessage('Hello')]
