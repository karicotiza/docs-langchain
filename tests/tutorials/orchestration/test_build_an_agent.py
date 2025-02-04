"""Test module.

Build an Agent.
"""

from typing import TYPE_CHECKING, Any

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from src.tutorials.orchestration.build_an_agent import (
    chat,
    search,
    stateful_agent_executor,
    stateless_agent_executor,
    tools,
)

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable


def test_search() -> None:
    """Test search tool."""
    message: str = "what is the weather in SF"
    expected_length: int = 250

    response: Any = search.invoke(message)

    assert len(response) > expected_length
    assert "snippet:" in response
    assert "San Francisco" in response


def test_model() -> None:
    """Test model."""
    message: str = "hi!"
    response: BaseMessage = chat.invoke([HumanMessage(message)])

    assert response.content == "How can I assist you today?"


def test_tool_call() -> None:
    """Test tool call."""
    message_1: str = "Hi!"
    message_2: str = "What's the weather in SF?"
    model_with_tools: Runnable = chat.bind_tools(tools)

    response_1: Any = model_with_tools.invoke([HumanMessage(message_1)])

    assert response_1.content == ""
    assert response_1.tool_calls[0]["args"]["query"] == "current events"
    assert response_1.tool_calls[0]["name"] == "duckduckgo_results_json"
    assert response_1.tool_calls[0]["type"] == "tool_call"

    response_2: Any = model_with_tools.invoke([HumanMessage(message_2)])

    assert response_2.content == ""
    assert response_2.tool_calls[0]["args"]["query"] == "SF weather"
    assert response_2.tool_calls[0]["name"] == "duckduckgo_results_json"
    assert response_2.tool_calls[0]["type"] == "tool_call"


def test_stateless_agent_1() -> None:
    """Test stateless agent."""
    message: str = "hi!"
    response: dict[str, Any] | Any = stateless_agent_executor.invoke(
        input={
            "messages": [HumanMessage(message)],
        },
    )

    assert "news" in response["messages"][-1].content.lower()


def test_stateless_agent_2() -> None:
    """Test stateless agent."""
    message: str = "whats the weather in sf?"
    response: dict[str, Any] | Any = stateless_agent_executor.invoke(
        input={
            "messages": [HumanMessage(message)],
        },
    )

    assert "current weather" in response["messages"][-1].content.lower()


def test_stateless_agent_streaming_messages() -> None:
    """Test stateless agent streaming messages."""
    message: str = "whats the weather in sf?"
    separator: str = "---"
    expected_length: int = 6
    memory: list[dict[str, Any] | Any] = []

    for chunk in stateless_agent_executor.stream(
        input={
            "messages": [HumanMessage(message)],
        },
    ):
        memory.append(chunk)
        memory.append(separator)

    assert len(memory) == expected_length
    assert memory[1::2] == [separator, separator, separator]


def test_stateful_agent() -> None:
    """Test stateful agent."""
    message_1: str = "hi im bob!"
    message_2: str = "whats my name?"
    config: RunnableConfig = RunnableConfig(
        {
            "configurable": {
                "thread_id": "123",
            },
        },
    )

    response_1 = stateful_agent_executor.invoke(
        input={
            "messages": [HumanMessage(message_1)],
        },
        config=config,
    )

    assert response_1["messages"][-1].content.startswith("Hi Bob!")

    response_2 = stateful_agent_executor.invoke(
        input={
            "messages": [HumanMessage(message_2)],
        },
        config=config,
    )

    assert response_2["messages"][-1].content.startswith("Hi Bob!")
