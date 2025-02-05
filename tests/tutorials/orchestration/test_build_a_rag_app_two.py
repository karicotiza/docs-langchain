"""Test module.

Build a Retrieval Augmented Generation (RAG) app: part 2.
"""

from typing import Any

from src.tutorials.orchestration.build_a_rag_app_two import graph


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
