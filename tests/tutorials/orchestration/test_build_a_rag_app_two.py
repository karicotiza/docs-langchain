"""Test module.

Build a Retrieval Augmented Generation (RAG) app: part 2.
"""

from src.tutorials.orchestration.build_a_rag_app_two import graph


def test_graph() -> None:
    """Test graph."""
    user_input_1: str = "Hello"

    response_1 = graph.invoke(
        input={"messages": [{"role": "user", "content": user_input_1}]},
    )

    assert response_1["messages"][-1].content == (
        "It seems like we've just started our conversation. "
        "I'm here to help answer any questions you may have. "
        "What's on your mind?"
    )
