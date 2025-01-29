"""Tutorial module.

Build an extraction chain.
"""

from typing import TYPE_CHECKING, Any

from src.tutorials.build_an_extraction_chain import (
    chat,
    message_no_extraction,
    messages,
    prompt_template,
    second_structured_llm,
    structured_llm,
    tool_messages,
)

if TYPE_CHECKING:
    from langchain_core.prompt_values import PromptValue


def test_extraction() -> None:
    """Test extraction."""
    text: str = "Alan Smith is 6 feet tall and has blond hair."
    prompt: PromptValue = prompt_template.invoke({"text": text})
    response: Any = structured_llm.invoke(prompt)

    assert response.name == "Alan Smith"
    assert response.hair_color == "blond"
    assert response.height in ["1.83", "1.8"]


def test_multiple_extraction() -> None:
    """Test multiple extraction."""
    text: str = (
        "My name is Jeff, my hair is black and i am 6 feet tall. "
        "Anna has the same color hair as me."
    )

    prompt: PromptValue = prompt_template.invoke({"text": text})
    response: Any = second_structured_llm.invoke(prompt)

    assert response.persons[0].name == "Jeff"
    assert response.persons[0].hair_color is None
    assert response.persons[0].height == "6 feet"

    assert response.persons[1].name == "Anna"
    assert response.persons[1].hair_color is None
    assert response.persons[1].height is None


def test_reference_examples() -> None:
    """Test reference examples."""
    response: Any = chat.invoke(messages)

    assert response.content == "6"

def test_tool_messages() -> None:
    """Test tool messages."""
    assert tool_messages[3].content == "Detected no people."
    assert tool_messages[7].content == "Detected people."


def test_performance() -> None:
    """Test performance."""
    # LLM can't handle multiple persons this time
    # response: Any = second_structured_llm.invoke([message_no_extraction])
    first_response: Any = structured_llm.invoke([message_no_extraction])

    assert first_response.name == "Earth"
    assert first_response.hair_color is None
    assert first_response.height is None

    second_response: Any = structured_llm.invoke(
        [*messages, message_no_extraction],
    )

    assert second_response.name == "Earth"
    assert second_response.hair_color is None
    assert second_response.height is None
