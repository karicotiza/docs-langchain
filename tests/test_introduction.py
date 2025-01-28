"""Test module.

Introduction.
"""


from typing import TYPE_CHECKING

from src.introduction import chat, message

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def test_invoke() -> None:
    """Test invoke method."""
    response: BaseMessage = chat.invoke(message)

    assert response.content == (
        "Hello! It's nice to meet you. "
        "Is there something I can help you with or would you like to chat?"
    )
