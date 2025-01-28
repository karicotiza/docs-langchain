"""Tutorial module.

Build a simple LLM application with chat models and prompt templates.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.settings import llm_model_name, llm_model_temperature, llm_model_url

chat: ChatOllama = ChatOllama(
    base_url=llm_model_url,
    model=llm_model_name,
    temperature=llm_model_temperature,
)

messages: list[HumanMessage | SystemMessage] = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

string_message: str = "Hello"
dict_messages: list[dict[str, str]] = [{"role": "user", "content": "Hello"}]
human_messages: list[HumanMessage] = [HumanMessage("Hello")]

prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", "Translate the following from English into {language}"),
    ("user", "{text}"),
])

language: str = "Italian"
text: str = "hi!"
