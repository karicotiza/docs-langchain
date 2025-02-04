"""Test module.

Build an extraction chain.

Notes:
* Sometimes Alan Smith's heigh is '1.83' and sometimes '1.8'.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import tool_example_to_messages
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict, Field

from src.settings import llm_model_name, llm_model_temperature, llm_model_url

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable


class Person(BaseModel):
    """Person model."""

    name: str | None = Field(
        default=None,
        description="The name of the person",
    )

    hair_color: str | None = Field(
        default=None,
        description="The color of the person's hair if known",
    )

    height: str | None = Field(
        default=None,
        description="Height measured in meters",
    )

    model_config = ConfigDict(
        coerce_numbers_to_str=True,
    )


system_prompt: tuple[str, str] = (
    "system",
    "You are an expert extraction algorithm. "
    "Only extract relevant information from the text. "
    "If you do not know the value of an attribute asked to extract, "
    "return null for the attribute's value.",
)

human_prompt: tuple[str, str] = ("human", "{text}")

prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [system_prompt, human_prompt],
)

chat: ChatOllama = ChatOllama(
    base_url=llm_model_url,
    model=llm_model_name,
    temperature=llm_model_temperature,
)

structured_llm: Runnable = chat.with_structured_output(Person)


class Persons(BaseModel):
    """Extracted data about people."""

    persons: list[Person]


second_structured_llm: Runnable = chat.with_structured_output(Persons)

messages: list[dict[str, str]] = [
    {"role": "user", "content": "2 🦜 2"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "2 🦜 3"},
    {"role": "assistant", "content": "5"},
    {"role": "user", "content": "3 🦜 4"},
]

tool_examples: list[tuple[str, Persons]] = [
    (
        "The ocean is vast and blue. It's more than 20,000 feet deep.",
        Persons(
            persons=[],
        ),
    ),
    (
        "Fiona traveled far from France to Spain.",
        Persons(
            persons=[
                Person(name="Fiona", height=None, hair_color=None),
            ],
        ),
    ),
]

tool_messages: list = []

for txt, tool_call in tool_examples:
    if tool_call.persons:
        ai_response: str = "Detected people."

    else:
        ai_response = "Detected no people."

    tool_messages.extend(
        tool_example_to_messages(txt, [tool_call], ai_response=ai_response),
    )

message_no_extraction: dict[str, str] = {
    "role": "user",
    "content": "The solar system is large, but earth has only 1 moon.",
}
