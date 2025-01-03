"""Test introduction module."""

from langchain_core.messages import AIMessage, BaseMessage

from src.introduction import chat


def test_chat_ollama_invoke() -> None:
    """Test ChatOllama invoke method."""
    response: BaseMessage = chat.invoke('Hi')

    assert isinstance(response, AIMessage)
    assert response.content == 'How can I assist you today?'
