"""Test module.

Build a chatbot.
"""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from src.tutorials.orchestration.build_a_chatbot import (
    app,
    chat,
    messages_to_trim,
    pirate_app,
    polyglot_app,
    trimmed_app,
)

pytest_plugins: tuple[str, ...] = ("pytest_asyncio",)


def _what_is_my_name() -> HumanMessage:
    return HumanMessage("What's my name?")


def _chunk_to_string(chunk: AIMessage) -> str:
    content: str | list[str | dict[Any, Any]] = chunk.content

    if isinstance(content, list):
        return "".join(str(content))

    return content


def _input(
    messages: list[BaseMessage],
    language: str | None = None,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "messages": messages,
    }

    if language:
        mapping["language"] = language

    return mapping


def _config(thread_id: str) -> RunnableConfig:
    return RunnableConfig(
        {
            "configurable": {
                "thread_id": thread_id,
            },
        },
    )


def _answer(response: dict[str, Any] | Any) -> str:  # noqa: ANN401
    return response["messages"][-1].content


def test_model_without_memory() -> None:
    """Test model's without any memory."""
    response_1: BaseMessage = chat.invoke([HumanMessage("Hi, I'm Bob")])

    assert response_1.content == (
        "Hello Bob! It's nice to meet you. "
        "Is there something I can help you with or would you like to chat?"
    )

    response_2: BaseMessage = chat.invoke([_what_is_my_name()])

    assert response_2.content == (
        "I don't have any information about your identity, "
        "so I'm not aware of your name. We just started our conversation, "
        "and I don't retain any data about individual users. "
        "If you'd like to share your name with me, "
        "I'd be happy to chat with you!"
    )


def test_model_with_manual_memory() -> None:
    """Test model's memory passing history manually."""
    response: BaseMessage = chat.invoke(
        [
            HumanMessage("Hi, I'm Bob"),
            AIMessage("Hello Bob! How can I assist you today?"),
            _what_is_my_name(),
        ],
    )

    assert response.content == (
        "Your name is Bob. We just established that earlier. "
        "Is there something specific you'd like to talk about or ask about, "
        "Bob?"
    )


def test_app_memory() -> None:
    """Test app's memory."""
    response_1: dict[str, Any] | Any = app.invoke(
        _input([HumanMessage("Hi! I'm Bob.")]),
        _config("abc123"),
    )

    assert _answer(response_1) == (
        "Hello Bob! It's nice to meet you. "
        "Is there something I can help you with, or would you like to chat?"
    )

    response_2: dict[str, Any] | Any = app.invoke(
        _input([_what_is_my_name()]),
        _config("abc123"),
    )

    assert _answer(response_2) == (
        "Your name is Bob. You told me that earlier."
    )

    response_3: dict[str, Any] | Any = app.invoke(
        _input([_what_is_my_name()]),
        _config("abc234"),
    )

    assert _answer(response_3) == (
        "I don't have any information about your identity, "
        "so I'm not aware of your name. We just started our conversation, "
        "and I don't retain any data about individual users. "
        "If you'd like to share your name with me, "
        "I'd be happy to chat with you!"
    )

    response_4: dict[str, Any] | Any = app.invoke(
        _input([_what_is_my_name()]),
        _config("abc123"),
    )

    assert _answer(response_4) == (
        """You didn't tell me your last name, just "Bob". """
        "I don't have any additional information about you beyond that. "
        "Would you like to share it with me?"
    )


@pytest.mark.asyncio
async def test_app_memory_async() -> None:
    """Test app's memory async."""
    response_1: dict[str, Any] | Any = await app.ainvoke(
        _input([HumanMessage("Hi! I'm Bob.")]),
        _config("abc111"),
    )

    assert _answer(response_1) == (
        "Hello Bob! It's nice to meet you. "
        "Is there something I can help you with, or would you like to chat?"
    )

    response_2: dict[str, Any] | Any = await app.ainvoke(
        _input([_what_is_my_name()]),
        _config("abc111"),
    )

    assert _answer(response_2) == (
        "Your name is Bob. You told me that earlier."
    )

    response_3: dict[str, Any] | Any = await app.ainvoke(
        _input([_what_is_my_name()]),
        _config("abc222"),
    )

    assert _answer(response_3) == (
        "I don't have any information about your identity, "
        "so I'm not aware of your name. We just started our conversation, "
        "and I don't retain any data about individual users. "
        "If you'd like to share your name with me, "
        "I'd be happy to chat with you!"
    )

    response_4: dict[str, Any] | Any = await app.ainvoke(
        _input([_what_is_my_name()]),
        _config("abc111"),
    )

    assert _answer(response_4) == (
        """You didn't tell me your last name, just "Bob". """
        "I don't have any additional information about you beyond that. "
        "Would you like to share it with me?"
    )


def test_pirate_app() -> None:
    """Test pirate app."""
    message_1: str = "Hi! I'm Jim."

    response_1: dict[str, Any] | Any = pirate_app.invoke(
        _input([HumanMessage(message_1)]),
        _config("abc345"),
    )

    assert _answer(response_1) == (
        "Yer lookin' fer a chat, eh? Well, matey Jim, welcome aboard! "
        "I be happy to have ye as me guest. "
        "What be bringin' ye to these fair waters today? "
        "Got any questions or just want to spin some yarns with ol' "
        "Blackbeak Betty?"
    )

    response_2: dict[str, Any] | Any = pirate_app.invoke(
        _input([_what_is_my_name()]),
        _config("abc345"),
    )

    assert _answer(response_2) == (
        "Arrr, ye be askin' about yer own name, eh? Yer name be Jim, matey! "
        "I remember now. "
        "Ye told me that when we set sail fer this here conversation. "
        "So, Jim, what be on yer mind? "
        "Want to talk about the seven seas or maybe find treasure?"
    )


def test_polyglot_app() -> None:
    """Test polyglot app."""
    message_1: str = "Hi! I'm Bob."
    message_2: str = "What is my name?"
    language: str = "Spanish"

    response_1: dict[str, Any] | Any = polyglot_app.invoke(
        _input([HumanMessage(message_1)], language),
        _config("abc456"),
    )

    assert _answer(response_1) == (
        "Hola Bob, ¿cómo estás? (Hello Bob, how are you?)"
    )

    response_2: dict[str, Any] | Any = polyglot_app.invoke(
        _input([HumanMessage(message_2)], language),
        _config("abc456"),
    )

    assert _answer(response_2) == (
        "Tu nombre es Bob. (Your name is Bob.)"
    )


def test_trimmed_app_1() -> None:
    """Test trimmed app."""
    language: str = "English"

    response: dict[str, Any] | Any = trimmed_app.invoke(
        _input([*messages_to_trim, _what_is_my_name()], language),
        _config("abc567"),
    )

    assert _answer(response) == (
        "I don't know your name. We just started our conversation, "
        "and I don't have any information about you. "
        "Would you like to tell me your name?"
    )


def test_trimmed_app_2() -> None:
    """Test trimmed app."""
    message: str = "What math problem did I ask?"
    language: str = "English"

    response: dict[str, Any] | Any = trimmed_app.invoke(
        _input([*messages_to_trim, HumanMessage(message)], language),
        _config("abc678"),
    )

    assert _answer(response) == 'You asked me "whats 2 + 2".'


def test_app_stream() -> None:
    """Test app's stream."""
    message: str = "Hi I'm Todd, please tell me a joke."
    language: str = "English"
    memory: list[str] = []

    for chunk, _ in trimmed_app.stream(
        _input([HumanMessage(message)], language),
        _config("abc789"),
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):
            memory.append(_chunk_to_string(chunk))
            memory.append("|")

    assert "".join(memory) == (
        "Hi| Todd|!| Here|'s| one| for| you|:\n\n"
        "|What| do| you| call| a| fake| nood|le|?\n\n"
        "|(wait| for| it|...)\n\n"
        "|An| imp|asta|!\n\n"
        "|Hope| that| made| you| smile|!| "
        "Do| you| want| to| hear| another| one|?||"
    )
