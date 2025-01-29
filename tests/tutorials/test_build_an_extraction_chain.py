"""Tutorial module.

Build an extraction chain.
"""

from typing import TYPE_CHECKING, Any

from src.tutorials.build_an_extraction_chain import (
    prompt_template,
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
    assert str(response.height) == "1.83"
