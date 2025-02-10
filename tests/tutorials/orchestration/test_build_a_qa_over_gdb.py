"""Test module.

Build a Question Answering application over a Graph Database.
"""

from src.tutorials.orchestration.build_a_qa_over_gdb import (
    graph,
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
        "Relationship properties:\n\n"
        "The relationships:\n"
        "(:Movie)-[:IN_GENRE]->(:Genre)\n"
        "(:Person)-[:DIRECTED]->(:Movie)\n"
        "(:Person)-[:ACTED_IN]->(:Movie)"
    )
