"""Test build a semantic search engine module."""

from src.tutorials.build_a_semantic_search_engine import docs


def test_docs_length() -> None:
    """Test docs length"""
    reference_length: int = 107

    assert len(docs) == reference_length
