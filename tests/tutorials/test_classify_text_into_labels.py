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

    assert response.model_dump() == {
        # "sentiment": "angry",
        "sentiment": "negative",
        # "aggressiveness": 10,
        "aggressiveness": 9,
        "language": "Spanish",
    }


def test_finer_structured_output_1() -> None:
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


def test_finer_structured_output_2() -> None:
    """Test runnable's with structured output method with finer model."""
    inp: str = "Estoy muy enojado con vos! Te voy a dar tu merecido!"

    prompt: PromptValue = finer_tagging_prompt.invoke({"input": inp})
    response: Any = finer_llm.invoke(prompt)

    assert response.sentiment == SentimentChoice.sad
    # assert response.aggressiveness == AggressivenessChoice.eight
    assert response.aggressiveness == AggressivenessChoice.nine
    assert response.language == LanguageChoice.spanish


def test_finer_structured_output_3() -> None:
    """Test runnable's with structured output method with finer model."""
    inp: str = (
        "Weather is ok here, I can go outside without much more than a coat"
    )

    prompt: PromptValue = finer_tagging_prompt.invoke({"input": inp})
    response: Any = finer_llm.invoke(prompt)

    assert response.sentiment == SentimentChoice.neutral
    assert response.aggressiveness == AggressivenessChoice.three
    assert response.language == LanguageChoice.english
