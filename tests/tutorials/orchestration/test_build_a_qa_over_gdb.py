"""Test module.

Build a Question Answering application over a Graph Database.
"""

from typing import Any

from src.tutorials.orchestration.build_a_qa_over_gdb import (
    enhanced_graph,
    graph,
    langgraph,
    llama_chain,
    phi4_chain,
)


def test_graph_schema() -> None:
    """Test graph schema."""
    graph.refresh_schema()

    assert graph.schema == (
        "Node properties:\n"
        "Movie "
        "{imdbRating: FLOAT, id: STRING, released: DATE, title: STRING}\n"
        "Person {name: STRING}\n"
        "Genre {name: STRING}\n"
        "Chunk {query: STRING, embedding: LIST, id: STRING, "
        "question: STRING, text: STRING}\n"
        "Relationship properties:\n\n"
        "The relationships:\n"
        "(:Movie)-[:IN_GENRE]->(:Genre)\n"
        "(:Person)-[:DIRECTED]->(:Movie)\n"
        "(:Person)-[:ACTED_IN]->(:Movie)"
    )


def test_enhanced_graph_schema() -> None:
    """Test enhanced graph schema."""
    enhanced_graph.refresh_schema()

    assert enhanced_graph.schema.startswith(
        "Node properties:\n"
        "- **Movie**\n"
        "  - `imdbRating`: FLOAT Min: 2.4, Max: 9.3\n"
        '  - `id`: STRING Example: "1"\n'
        "  - `released`: DATE Min: 1964-12-16, Max: 1996-09-15\n"
        '  - `title`: STRING Example: "Toy Story"\n'
        "- **Person**\n"
        '  - `name`: STRING Example: "John Lasseter"\n'
        "- **Genre**\n"
        '  - `name`: STRING Example: "Adventure"\n'
    )


# LLaMa3.2:3B can't handle Cypher requests
def test_graph_cypher_qa_chain_1() -> None:
    """Test graph cypher qa chain 1."""
    user_input: str = "What was the cast of the Casino?"
    response: dict[str, Any] = llama_chain.invoke({"query": user_input})

    assert response["result"] == (
        "I don't have enough information to provide an accurate answer."
    )


# Phi4 can handle Cypher requests
def test_graph_cypher_qa_chain_2() -> None:
    """Test graph cypher qa chain 2."""
    user_input: str = "What was the cast of the Casino?"
    response: dict[str, Any] = phi4_chain.invoke({"query": user_input})

    assert response["result"] == (
        'The cast of "Casino" included Joe Pesci, Sharon Stone, '
        "Robert De Niro, and James Woods."
    )


# But Phi4 can't handle function calling
def test_lang_graph() -> None:
    """Test langgraph."""
    response: Any = langgraph.invoke(
        input={"question": "What's the weather in Spain?"},
    )

    assert response == {
        "answer": (
            "I'm unable to provide current weather information for Spain "
            "as my knowledge cutoff is December 2023, "
            "and I don't have real-time access to current data. However, "
            "I can suggest checking a reliable weather website or app, "
            "such as AccuWeather or BBC Weather, "
            "for the most up-to-date forecast."
        ),
        "steps": ["guardrail", "generate_final_answer"],
    }
