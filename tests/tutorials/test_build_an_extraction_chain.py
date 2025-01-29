"""Tutorial module.

Build an extraction chain.
"""

from typing import TYPE_CHECKING, Any

from src.tutorials.build_an_extraction_chain import (
    prompt_template,
    second_structured_llm,
    structured_llm,
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
    assert response.height == "1.83"


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
