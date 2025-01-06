"""Test build a semantic search engine module."""

import pytest
from langchain_core.documents import Document

from src.tutorials.build_a_semantic_search_engine import (
    all_splits,
    docs,
    first_vector,
    second_vector,
    vector_store,
)

pytest_plugins: tuple[str, ...] = (
    'pytest_asyncio',
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


def test_vector_store() -> None:
    """Test vector store."""
    results: list[Document] = vector_store.similarity_search(
        "How many distribution centers does Nike have in the US?"
    )

    result_reference: str = ' '.join((
        'operations. We also lease an office complex in Shanghai, China,',
        'our headquarters for our Greater China geography, occupied by',
        'employees focused on implementing our\nwholesale, NIKE Direct',
        'and merchandising strategies in the region, among other',
        'functions.\nIn the United States, NIKE has eight significant',
        'distribution centers. Five are located in or near Memphis,',
        'Tennessee, two of which are owned and three of which are\nleased.',
        'Two other distribution centers, one located in Indianapolis,',
        'Indiana and one located in Dayton, Tennessee, are leased and',
        'operated by third-party logistics\nproviders. One distribution',
        'center for Converse is located in Ontario, California, which is',
        'leased. NIKE has a number of distribution facilities outside the',
        'United States,\nsome of which are leased and operated by third-party',
        'logistics providers. The most significant distribution',
        'facilities outside the United States are located in Laakdal,',
    ))

    assert results[0].page_content == result_reference


@pytest.mark.asyncio
async def test_async_vector_store() -> None:
    """Test async vector store."""
    results: list[Document] = await vector_store.asimilarity_search(
        'When was Nike incorporated?'
    )

    result_reference: str = ''.join((
        'Table of Contents\nPART I\nITEM 1. BUSINESS\nGENERAL\nNIKE, Inc. ',
        'was incorporated in 1967 under the laws of the State of Oregon. As ',
        'used in this Annual Report on Form 10-K (this "Annual Report"), the ',
        'terms "we," "us," "our,"\n"NIKE" and the "Company" refer to NIKE, ',
        'Inc. and its predecessors, subsidiaries and affiliates, ',
        'collectively, unless the context indicates otherwise.\nOur ',
        'principal business activity is the design, development and ',
        'worldwide marketing and selling of athletic footwear, apparel, ',
        'equipment, accessories and services. NIKE is\nthe largest seller of ',
        'athletic footwear and apparel in the world. We sell our products ',
        'through NIKE Direct operations, which are comprised of both ',
        'NIKE-owned retail stores\nand sales through our digital platforms ',
        '(also referred to as "NIKE Brand Digital"), to retail accounts and ',
        'to a mix of independent distributors, licensees and sales',
    ))

    assert results[0].page_content == result_reference
