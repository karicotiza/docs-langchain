"""Test module.

Build a Retrieval Augmented Generation (RAG) app: part 2.
"""

from typing import Any

from src.tutorials.orchestration.build_a_rag_app_two import (
    agent,
    graph,
    stateful_graph,
)

def test_graph_1() -> None:
    """Test graph 1."""
    user_input: str = "Hello"

    expected_output_1: str = (
        "It seems like we've just started our conversation. "
        "I'm here to help answer any questions you may have. "
        "What's on your mind?"
    )

    expected_output_2: str = (
        "It seems like we've started a new conversation. "
        "I'm ready to help answer any questions you may have. "
        "What's on your mind?"
    )

    response: dict[str, Any] | Any = graph.invoke(
        input={"messages": [{"role": "user", "content": user_input}]},
    )

    assert response["messages"][-1].content in [
        expected_output_1, expected_output_2,
    ]


def test_graph_2() -> None:
    """Test graph 2."""
    user_input: str = "What is Task Decomposition?"

    response: dict[str, Any] | Any = graph.invoke(
        input={"messages": [{"role": "user", "content": user_input}]},
    )

    expected_output_1: str = (
        "Task decomposition is the process of breaking down a complicated "
        "task into smaller and simpler steps. "
        "This technique helps an agent plan ahead by identifying individual "
        "subgoals that can be achieved one at a time. "
        "It transforms big tasks into multiple manageable tasks, "
        "allowing for more efficient problem-solving."
    )

    expected_output_2: str = (
        "Task decomposition is a technique used to break down complex "
        "tasks into smaller and simpler steps. "
        "This process helps an agent understand the task and plan ahead by "
        "identifying subgoals and manageable steps. "
        "It transforms big tasks into multiple manageable tasks, "
        "allowing for more efficient computation and interpretation of "
        "the model's thinking process."
    )

    assert response["messages"][-1].content in [
        expected_output_1, expected_output_2,
    ]


def test_stateful_graph() -> None:
    """Test stateful graph."""
    user_input_1: str = "What is Task Decomposition?"
    user_input_2: str = "Can you look up some common ways of doing it?"

    response_1: dict[str, Any] | Any = stateful_graph.invoke(
        input={"messages": [{"role": "user", "content": user_input_1}]},
        config={"configurable": {"thread_id": "abc123"}},
    )

    expected_output_1_1: str = (
        "Task decomposition is the process of breaking down a complicated "
        "task into smaller and simpler steps. "
        "This technique helps an agent plan ahead by identifying individual "
        "subgoals that can be achieved one at a time. "
        "It transforms big tasks into multiple manageable tasks, "
        "allowing for more efficient problem-solving."
    )

    expected_output_1_2: str = (
        "Task decomposition is a technique used to break down complex "
        "tasks into smaller and simpler steps. "
        "This process helps an agent understand the task and plan ahead by "
        "identifying subgoals and manageable steps. "
        "It transforms big tasks into multiple manageable tasks, "
        "allowing for more efficient computation and interpretation of "
        "the model's thinking process."
    )

    expected_output_2_1: str = "Task decomposition can be done in three"

    assert response_1["messages"][-1].content in [
        expected_output_1_1, expected_output_1_2,
    ]

    response_2: dict[str, Any] | Any = stateful_graph.invoke(
        input={"messages": [{"role": "user", "content": user_input_2}]},
        config={"configurable": {"thread_id": "abc123"}},
    )

    assert expected_output_2_1 in response_2["messages"][-1].content


def test_agent() -> None:
    """Test agent."""
    user_input: str = (
        "What is the standard method for Task Decomposition?\n\n"
        "Once you get the answer, look up common extensions of that method."
    )

    expected_output: str = "Common extensions of this method include" 

    response: dict[str, Any] | Any = agent.invoke(
        input={"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": "def234"}},
    )

    assert expected_output in response["messages"][-1].content
