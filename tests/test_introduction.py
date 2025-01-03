"""Test introduction module."""

from langchain_core.messages import AIMessage, BaseMessage

from src.introduction import chat, message


def test_chat_ollama_invoke() -> None:
    """Test ChatOllama invoke method."""
    response: BaseMessage = chat.invoke(message)

    assert isinstance(response, AIMessage)
    assert response.content == (
        "Hello! It's nice to meet you. " +
        "Is there something I can help you with or would you like to chat?"
    )
