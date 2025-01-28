"""Test module.

Classify text into labels.
"""

from typing import TYPE_CHECKING, Any

from src.tutorials.classify_text_into_labels import (
    Classification,
    llm,
    tagging_prompt,
)

if TYPE_CHECKING:
    from langchain_core.prompt_values import PromptValue


def test_structured_output() -> None:
    """Test runnable's with structured output method."""
    inp: str = (
        "Estoy increiblemente contento de haberte conocido! "
        "Creo que seremos muy buenos amigos!"
    )

    prompt: PromptValue = tagging_prompt.invoke({"input": inp})
    response: Any = llm.invoke(prompt)

    assert response == Classification(
        sentiment="positive",
        aggressiveness=1,
        language="Spanish",
    )


def test_dictionary_output() -> None:
    """Test response's dict method."""
    inp: str = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
    prompt: PromptValue = tagging_prompt.invoke({"input": inp})
    response: Any = llm.invoke(prompt)

    assert response.dict() == {
        "sentiment": "angry",
        "aggressiveness": 1,
        "language": "Spanish",
    }
