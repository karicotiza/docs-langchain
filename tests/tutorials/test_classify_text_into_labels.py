"""Test module.

Classify text into labels.
"""

from typing import TYPE_CHECKING, Any

from src.tutorials.classify_text_into_labels import (
    AggressivenessChoice,
    LanguageChoice,
    SentimentChoice,
    finer_llm,
    finer_tagging_prompt,
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

    assert response.sentiment == "positive"
    assert response.aggressiveness == 1
    assert response.language == "Spanish"


def test_dictionary_output() -> None:
    """Test response's dict method."""
    inp: str = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
    prompt: PromptValue = tagging_prompt.invoke({"input": inp})
    response: Any = llm.invoke(prompt)

    assert response.dict() == {
        "sentiment": "negative",
        "aggressiveness": 9,
        "language": "Spanish",
    }


def test_finer_structured_output() -> None:
    """Test runnable's with structured output method with finer model."""
    inp: str = (
        "Estoy increiblemente contento de haberte conocido! "
        "Creo que seremos muy buenos amigos!"
    )

    prompt: PromptValue = finer_tagging_prompt.invoke({"input": inp})
    response: Any = finer_llm.invoke(prompt)

    assert response.sentiment == SentimentChoice.happy
    assert response.aggressiveness == AggressivenessChoice.two
    assert response.language == LanguageChoice.spanish
