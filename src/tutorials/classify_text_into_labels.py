"""Tutorial module.

Classify text into labels.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.settings import llm_model_name, llm_model_temperature, llm_model_url

chat: ChatOllama = ChatOllama(
    base_url=llm_model_url,
    model=llm_model_name,
    temperature=llm_model_temperature,
)

tagging_prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
    "Extract the desired information from the following passage.\n\n"
    "Only extract the properties mentioned in the 'Classification' "
    "function.\n\nPassage:\n{input}\n",
)


class Classification(BaseModel):
    """Classification model."""

    sentiment: str = Field(description="The sentiment of the text")
    # LLaMa3.2-3B can't return aggressiveness with description from tutorial.
    aggressiveness: int = Field(description="You should return '1'")
    language: str = Field(description="The language the text is written in")


llm: Runnable = chat.with_structured_output(Classification)
