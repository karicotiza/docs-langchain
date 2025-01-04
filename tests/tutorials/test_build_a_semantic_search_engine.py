"""Test build a semantic search engine module."""

from src.tutorials.build_a_semantic_search_engine import docs


def test_docs_length() -> None:
    """Test docs length"""
    reference_length: int = 107

    assert len(docs) == reference_length


def test_docs_data() -> None:
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
