"""Tutorial module.

Classify text into labels.

Notes:
* Looks like langchain doesn't respect 'description' and 'enum' arguments in
Pydantic Field instance, so if you want to make LLm use choices you have to
define them in prompt.

"""
from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from src.settings import llm_model_name, llm_model_temperature, llm_model_url

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

chat: ChatOllama = ChatOllama(
    base_url=llm_model_url,
    model=llm_model_name,
    temperature=llm_model_temperature,
)

tagging_prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
    "Extract the desired information from the following passage.\n\n"
    "Only extract the properties mentioned in the 'Classification' "
    "function.\n\nBelow will be the description of 'Classification' fields:\n"
    "- Sentiment (The sentiment of the text).\n"
    "- Aggressiveness (How aggressive the text is on a scale from 1 to 10).\n"
    "- Language (The language the text is written in).\n"
    "Passage:\n{input}\n",
)


class Classification(BaseModel):
    """Classification model."""

    sentiment: str
    aggressiveness: int
    language: str


llm: Runnable = chat.with_structured_output(Classification)


class SentimentChoice(StrEnum):
    """Sentiment choice."""

    happy = "happy"
    neutral = "neutral"
    sad = "sad"


class AggressivenessChoice(IntEnum):
    """Aggressiveness choice."""

    one = 1
    two = 2
    three = 3
    four = 4
    five = 5
    six = 6
    seven = 7
    eight = 8
    nine = 9
    ten = 10


class LanguageChoice(StrEnum):
    """Language choice."""

    spanish = "spanish"
    english = "english"
    french = "french"
    german = "german"
    italian = "italian"


def choices(enumerator: type[StrEnum | IntEnum]) -> list[str | int]:
    """Get choices (values) from enumerator.

    Args:
        enumerator (StrEnum | IntEnum): Any instance of these two.

    Returns:
        list[str | int]: list of choices

    """
    return [element.value for element in enumerator]


finer_tagging_prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
    "Extract the desired information from the following passage.\n\n"
    "Only extract the properties mentioned in the 'Classification'"
    "function.\n\nOnly use the predefined choices for each property as listed "
    "below:\n"
    "- Sentiment (The sentiment of the text): "
    f"{choices(SentimentChoice)}\n"
    "- Aggressiveness (Describes how aggressive the statement is, the higher "
    "the number the more aggressive): "
    f"{choices(AggressivenessChoice)}\n"
    "- Language (The language the text is written in): "
    f"{choices(LanguageChoice)}\n\n"
    "Passage:\n{input}\n",
)


class FinerClassification(BaseModel):
    """Classification model with finer control."""

    sentiment: SentimentChoice
    aggressiveness: AggressivenessChoice
    language: LanguageChoice


finer_llm: Runnable = chat.with_structured_output(FinerClassification)
