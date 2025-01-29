"""Test module.

Build an extraction chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

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

    height: float | None = Field(
        default=None,
        description="Height measured in meters",
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
