"""Test module.

Build a Retrieval Augmented Generation (RAG) app: part 1.
"""

from typing import Any

import pytest

from src.tutorials.orchestration.build_a_rag_app_one import (
    all_splits,
    docs,
    graph,
    prompt,
)

# def test_rag_app() -> None:
#     """Test rag app."""
#     user_input: dict[str, str] = {"question": "What is Task Decomposition?"}
#     response: dict[str, Any] | Any = graph.invoke(user_input)
#     answer: str = response["answer"]

#     assert answer == (
#         "Task Decomposition is a technique used to break down complex tasks "
#         "into smaller, simpler steps. It involves instructing a model "
#         'to "think step by step" and utilize more computation to decompose '
#         "hard tasks into manageable subgoals. "
#         "This can be done through various prompting techniques, "
#         "such as Chain of Thought or Tree of Thoughts, "
#         "which generate multiple reasoning possibilities at each step."
#     )


# def test_web_base_loader() -> None:
#     """Test WebBaseLoader."""
#     expected_documents: int = 1
#     expected_characters: int = 43130
#     slice_size: int = 500

#     assert len(docs) == expected_documents
#     assert len(docs[0].page_content) == expected_characters
#     assert docs[0].page_content[:slice_size] == (
#         "\n\n      LLM Powered Autonomous Agents\n    \n"
#         "Date: June 23, 2023  |  Estimated Reading Time: 31 min  |  "
#         "Author: Lilian Weng\n\n\n"
#         "Building agents with LLM (large language model) as its core "
#         "controller is a cool concept. Several proof-of-concepts demos, "
#         "such as AutoGPT, GPT-Engineer and BabyAGI, "
#         "serve as inspiring examples. The potentiality of LLM extends beyond "
#         "generating well-written copies, stories, essays and programs; "
#         "it can be framed as a powerful general problem solver.\n"
#         "Agent System Overview#\n"
#         "In"
#     )


# def test_recursive_character_text_splitter() -> None:
#     """Test RecursiveCharacterTextSplitter."""
#     expected_documents: int = 66
#     assert len(all_splits) == expected_documents


# def test_prompt() -> None:
#     """Test prompt."""
#     system_input: dict[str, str] = {
#         "context": "(context)",
#         "question": "(question)",
#     }

#     response: Any = prompt.invoke(system_input).to_messages()

#     assert response[0].content == (
#         "You are an assistant for question-answering tasks. "
#         "Use the following pieces of retrieved context to answer the "
#         "question. If you don't know the answer, "
#         "just say that you don't know. "
#         "Use three sentences maximum and keep the answer concise.\n"
#         "Question: (question) \n"
#         "Context: (context) \n"
#         "Answer:"
#     )


@pytest.mark.asyncio
async def test_graph_stream() -> None:
    """Test graph's stream."""
    user_input: dict[str, str] = {"question": "What is Task Decomposition?"}
    memory: list[str] = []

    for message, _ in graph.stream(user_input, stream_mode="messages"):
        memory.append(message.content)

    assert "|".join(memory) == (
        "Task| Decom|position| is| a| technique| used| to| break| down| "
        "complex| tasks| into| smaller|,| simpler| steps|.| It| involves| "
        'instruct|ing| a| model| to| "|think| step| by| step|"| and| utilize| '
        "more| computation| to| decom|pose| hard| tasks| into| manageable| "
        "sub|goals|.| This| can| be| done| through| various| prompting| "
        "techniques|,| such| as| Chain| of| Thought| or| Tree| of| "
        "Thoughts|,| which| generate| multiple| reasoning| possibilities| at| "
        "each| step|.|"
    )
