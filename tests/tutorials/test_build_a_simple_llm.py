"""Test build a simple LLM application with chat models and prompt templates
module."""

from langchain_core.messages import BaseMessage

from src.tutorials.build_a_simple_llm import (
    chat,
    messages,
    string_message,
    dict_messages,
    human_messages,
)


def test_invoke_with_messages() -> None:
    """Test ChatOllama invoke method with messages as args."""
    response: BaseMessage = chat.invoke(messages)

    assert response.content == 'Ciao! Come posso aiutarti oggi?'


def test_support_for_different_input_formats() -> None:
    """Test ChatOllama invoke method different input format support"""
    answer: str = 'How can I assist you today?'

    assert chat.invoke(string_message).content == answer
    assert chat.invoke(dict_messages).content == answer
    assert chat.invoke(human_messages).content == answer
