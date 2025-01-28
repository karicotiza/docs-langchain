"""Test module.

Build a simple LLM application with chat models and prompt templates.
"""


from typing import TYPE_CHECKING

from src.tutorials.build_a_simple_llm import (
    chat,
    dict_messages,
    human_messages,
    language,
    messages,
    prompt_template,
    string_message,
    text,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.messages import BaseMessage, BaseMessageChunk
    from langchain_core.prompt_values import PromptValue


def test_invoke_with_messages() -> None:
    """Test invoke method with messages."""
    response: BaseMessage = chat.invoke(messages)

    assert response.content == "Ciao! Come posso aiutarti oggi?"


def test_support_for_different_input_formats() -> None:
    """Test support for different input formats."""
    answer: str = "How can I assist you today?"

    assert chat.invoke(string_message).content == answer
    assert chat.invoke(dict_messages).content == answer
    assert chat.invoke(human_messages).content == answer


def test_stream_with_messages() -> None:
    """Test stream method with messages."""
    memory: str = ""
    stream: Iterator[BaseMessageChunk] = chat.stream(messages)

    for token in stream:
        memory = "|".join((memory, str(token.content)))

    assert memory == "|C|iao|!| Come| pos|so| ai|ut|arti| oggi|?|"


def test_invoke_template() -> None:
    """Test invoke method on template."""
    prompt: PromptValue = prompt_template.invoke(
        {"language": language, "text": text},
    )

    assert prompt.to_messages() == messages

    response: BaseMessage = chat.invoke(prompt)

    assert response.content == "Ciao! Come posso aiutarti oggi?"
