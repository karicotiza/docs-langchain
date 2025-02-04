"""Test module.

Build a Retrieval Augmented Generation (RAG) app: part 1.
"""

from typing import TYPE_CHECKING, Any

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from src.tutorials.orchestration.build_a_rag_app_one import graph


def test_rag_app() -> None:
    """Test rag app."""
    user_input: dict[str, str] = {"question": "What is Task Decomposition?"}
    response: dict[str, Any] | Any = graph.invoke(user_input)
    answer: str = response["answer"]

    assert answer == (
        "Task Decomposition is a technique used to break down complex tasks "
        "into smaller, simpler steps. It involves instructing a model "
        'to "think step by step" and utilize more computation to decompose '
        "hard tasks into manageable subgoals. "
        "This can be done through various prompting techniques, "
        "such as Chain of Thought or Tree of Thoughts, "
        "which generate multiple reasoning possibilities at each step."
    )
