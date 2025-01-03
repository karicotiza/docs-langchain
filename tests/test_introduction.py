"""Test introduction module."""

from langchain_core.messages import BaseMessage

from src.introduction import chat, message


def test_invoke() -> None:
    """Test invoke method."""
    response: BaseMessage = chat.invoke(message)

    assert response.content == (
        "Hello! It's nice to meet you. " +
        "Is there something I can help you with or would you like to chat?"
    )
