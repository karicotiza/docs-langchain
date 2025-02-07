"""Test module.

Summarize text.
"""

import pytest

from src.tutorials.orchestration.summarize_text import (
    app,
    chain,
    docs,
    split_docs,
)

pytest_plugins: tuple[str, ...] = (
    "pytest_asyncio",
)


def test_chain() -> None:
    """Test chain."""
    expected_output: str = (
        "Based on the provided text, "
        "I will attempt to summarize the main points "
        "and identify potential areas for improvement."
    )

    assert chain.invoke(input={"context": docs}).startswith(expected_output)


def test_chain_streaming() -> None:
    """Test chain."""
    expected_output: list[str] = [
        "Based", " on", " the", " provided", " text", ",",
        " I", " will", " attempt", " to",
    ]

    assert list(chain.stream({"context": docs}))[:10] == expected_output


def test_character_text_splitter() -> None:
    """Test character text splitter."""
    expected_length: int = 14

    assert len(split_docs) == expected_length


@pytest.mark.asyncio
async def test_app() -> None:
    """Test app."""
    recursion_limit: int = 10
    expected_output: str = "consolidated summary of the main themes"
    steps: list = [
        step async for step in app.astream(
            {"contents": [doc.page_content for doc in split_docs]},
            {"recursion_limit": recursion_limit},
        )
    ]

    assert expected_output in (
        steps[-1]["generate_final_summary"]["final_summary"]
    )