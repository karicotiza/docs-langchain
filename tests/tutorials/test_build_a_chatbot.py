"""Test module.

Build a chatbot.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from src.tutorials.build_a_chatbot import (
    app,
    chat,
    messages_to_trim,
    pirate_app,
    polyglot_app,
    trimmed_app,
)

pytest_plugins: tuple[str, ...] = ("pytest_asyncio",)


def test_model_without_memory() -> None:
    """Test model's without any memory."""
    response_1: BaseMessage = chat.invoke(
        [HumanMessage("Hi, I'm Bob")],
    )

    assert response_1.content == (
        "Hello Bob! It's nice to meet you. "
        "Is there something I can help you with or would you like to chat?"
    )

    response_2: BaseMessage = chat.invoke(
        [HumanMessage("What's my name?")],
    )

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
            HumanMessage("What's my name?"),
        ],
    )

    assert response.content == (
        "Your name is Bob. We just established that earlier. "
        "Is there something specific you'd like to talk about or ask about, "
        "Bob?"
    )


def test_app_memory() -> None:
    """Test app's memory."""
    config_1: RunnableConfig = RunnableConfig(
        {
            "configurable": {"thread_id": "abc123"},
        },
    )

    config_2: RunnableConfig = RunnableConfig(
        {
            "configurable": {"thread_id": "abc234"},
        },
    )

    response_1: dict[str, Any] | Any = app.invoke(
        {
            "messages": [HumanMessage("Hi! I'm Bob.")],
        },
        config_1,
    )

    assert response_1["messages"][-1].content == (
        "Hello Bob! It's nice to meet you. "
        "Is there something I can help you with, or would you like to chat?"
    )

    response_2: dict[str, Any] | Any = app.invoke(
        {
            "messages": [HumanMessage("What's my name?")],
        },
        config_1,
    )

    assert response_2["messages"][-1].content == (
        "Your name is Bob. You told me that earlier."
    )

    response_3: dict[str, Any] | Any = app.invoke(
        {
            "messages": [HumanMessage("What's my name?")],
        },
        config_2,
    )

    assert response_3["messages"][-1].content == (
        "I don't have any information about your identity, "
        "so I'm not aware of your name. We just started our conversation, "
        "and I don't retain any data about individual users. "
        "If you'd like to share your name with me, "
        "I'd be happy to chat with you!"
    )

    response_4: dict[str, Any] | Any = app.invoke(
        {
            "messages": [HumanMessage("What's my name?")],
        },
        config_1,
    )

    assert response_4["messages"][-1].content == (
        """You didn't tell me your last name, just "Bob". """
        "I don't have any additional information about you beyond that. "
        "Would you like to share it with me?"
    )


@pytest.mark.asyncio
async def test_app_memory_async() -> None:
    """Test app's memory async."""
    config_1: RunnableConfig = RunnableConfig(
        {
            "configurable": {"thread_id": "abc111"},
        },
    )

    config_2: RunnableConfig = RunnableConfig(
        {
            "configurable": {"thread_id": "abc222"},
        },
    )

    response_1: dict[str, Any] | Any = await app.ainvoke(
        {
            "messages": [HumanMessage("Hi! I'm Bob.")],
        },
        config_1,
    )

    assert response_1["messages"][-1].content == (
        "Hello Bob! It's nice to meet you. "
        "Is there something I can help you with, or would you like to chat?"
    )

    response_2: dict[str, Any] | Any = await app.ainvoke(
        {
            "messages": [HumanMessage("What's my name?")],
        },
        config_1,
    )

    assert response_2["messages"][-1].content == (
        "Your name is Bob. You told me that earlier."
    )

    response_3: dict[str, Any] | Any = await app.ainvoke(
        {
            "messages": [HumanMessage("What's my name?")],
        },
        config_2,
    )

    assert response_3["messages"][-1].content == (
        "I don't have any information about your identity, "
        "so I'm not aware of your name. We just started our conversation, "
        "and I don't retain any data about individual users. "
        "If you'd like to share your name with me, "
        "I'd be happy to chat with you!"
    )

    response_4: dict[str, Any] | Any = await app.ainvoke(
        {
            "messages": [HumanMessage("What's my name?")],
        },
        config_1,
    )

    assert response_4["messages"][-1].content == (
        """You didn't tell me your last name, just "Bob". """
        "I don't have any additional information about you beyond that. "
        "Would you like to share it with me?"
    )


def test_pirate_app() -> None:
    """Test pirate app."""
    config_1: RunnableConfig = RunnableConfig(
        {
            "configurable": {"thread_id": "abc345"},
        },
    )

    response_1: dict[str, Any] | Any = pirate_app.invoke(
        {
            "messages": [HumanMessage("Hi! I'm Jim.")],
        },
        config_1,
    )

    assert response_1["messages"][-1].content == (
        "Yer lookin' fer a chat, eh? Well, matey Jim, welcome aboard! "
        "I be happy to have ye as me guest. "
        "What be bringin' ye to these fair waters today? "
        "Got any questions or just want to spin some yarns with ol' "
        "Blackbeak Betty?"
    )

    response_2: dict[str, Any] | Any = pirate_app.invoke(
        {
            "messages": [HumanMessage("What's my name")],
        },
        config_1,
    )

    assert response_2["messages"][-1].content == (
        "Arrr, ye be askin' about yer own name, eh? Yer name be Jim, matey! "
        "Don't ye remember? I told ye that already, but I suppose ye were too "
        "busy swabbin' the decks to pay attention. "
        "Now, let's get back to more important things... "
        "like findin' the hidden treasure or avoidin' the Royal Navy!"
    )


def test_polyglot_app() -> None:
    """Test polyglot app."""
    config: RunnableConfig = RunnableConfig(
        {
            "configurable": {"thread_id": "abc456"},
        },
    )

    language: str = "Spanish"

    response_1: dict[str, Any] | Any = polyglot_app.invoke(
        {
            "messages": [HumanMessage("Hi! I'm Bob.")],
            "language": language,
        },
        config,
    )

    assert response_1["messages"][-1].content == (
        "Hola Bob, ¿cómo estás? (Hello Bob, how are you?)"
    )

    response_2: dict[str, Any] | Any = polyglot_app.invoke(
        {
            "messages": [HumanMessage("What is my name?")],
        },
        config,
    )

    assert response_2["messages"][-1].content == (
        "Tu nombre es Bob. (Your name is Bob.)"
    )


def test_trimmed_app_1() -> None:
    """Test trimmed app."""
    config: RunnableConfig = RunnableConfig(
        {
            "configurable": {"thread_id": "abc567"},
        },
    )

    language: str = "English"

    response: dict[str, Any] | Any = trimmed_app.invoke(
        {
            "messages": [*messages_to_trim, HumanMessage("What's my name?")],
            "language": language,
        },
        config,
    )

    assert response["messages"][-1].content == (
        "I don't know your name. We just started our conversation, "
        "and I don't have any information about you. "
        "Would you like to tell me your name?"
    )


def test_trimmed_app_2() -> None:
    """Test trimmed app."""
    config: RunnableConfig = RunnableConfig(
        {
            "configurable": {"thread_id": "abc678"},
        },
    )

    language: str = "English"

    response: dict[str, Any] | Any = trimmed_app.invoke(
        {
            "messages": [
                *messages_to_trim,
                HumanMessage("What math problem did I ask?"),
            ],
            "language": language,
        },
        config,
    )

    assert response["messages"][-1].content == (
        'You asked me "whats 2 + 2".'
    )
