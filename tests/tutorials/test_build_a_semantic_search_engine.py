"""Test build a semantic search engine module."""

from src.tutorials.build_a_semantic_search_engine import (
    all_splits,
    docs,
    first_vector,
    second_vector,
)


def test_docs_length() -> None:
    """Test docs length."""
    reference_length: int = 107

    assert len(docs) == reference_length


def test_docs_data() -> None:
    """Test docs data."""
    size_of_slice: int = 200
    reference_content: str = ''.join((
        'Table of Contents\n',
        'UNITED STATES\n',
        'SECURITIES AND EXCHANGE COMMISSION\n',
        'Washington, D.C. 20549\n',
        'FORM 10-K\n',
        '(Mark One)\n',
        '☑  ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(D) OF THE SECURITIES ',
        'EXCHANGE ACT OF 1934\n',
        'F',
    ))

    reference_metadata: dict[str, int | str] = {
        'page': 0, 'source': 'data/nke-10k-2023.pdf',
    }

    assert docs[0].page_content[:size_of_slice] == reference_content
    assert docs[0].metadata == reference_metadata


def test_all_splits_length() -> None:
    """Test all splits length."""
    reference_length: int = 516

    assert len(all_splits) == reference_length


def test_embedding_model() -> None:
    """Test embedding model."""
    reference_length: int = 1024
    reference_first_vector: list[float] = [
        -0.042318076,
        -0.024977539,
        -0.04551967,
        0.014961654,
        -0.0020620648,
        -0.019979082,
        0.0136758275,
        -0.0006364515,
        0.0006704848,
        0.04520819,
    ]

    assert len(first_vector) == len(second_vector)
    assert len(first_vector) == reference_length
    assert first_vector[:10] == reference_first_vector
