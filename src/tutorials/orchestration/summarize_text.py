"""Tutorial module.

Summarize text.
"""
from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import CharacterTextSplitter
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from src.settings import (
    llm_model_name,
    llm_model_temperature,
    llm_model_url,
)

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.prompt_values import PromptValue
    from langchain_core.runnables import Runnable
    from langgraph.graph.state import CompiledStateGraph

chat: ChatOllama = ChatOllama(
    base_url=llm_model_url,
    model=llm_model_name,
    temperature=llm_model_temperature,
)

url: str = "https://lilianweng.github.io/posts/2023-06-23-agent/"
loader: WebBaseLoader = WebBaseLoader(url)
docs: list[Document] = loader.load()

prompt_template: str = (
    "Write a concise summary of the following:\n\n"
    "{context}"
)

prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(prompt_template)
chain: Runnable[dict[str, Any], Any] = create_stuff_documents_chain(
    llm=chat,
    prompt=prompt,
)

map_prompt_value: str = (
    "Write a concise summary of the following:\n\n"
    "{context}"
)

map_prompt_url: str = "rlm/map-prompt"
map_prompt: ChatPromptTemplate = hub.pull(map_prompt_url)

reduce_template: str = (
    "The following is a set of summaries:\n\n"
    "{docs}\n\n"
    "Take these and distill it into a final, consolidated summary "
    "of the main themes."
)

reduce_prompt = ChatPromptTemplate.from_template(reduce_template)

splitter: CharacterTextSplitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=0,
)

split_docs: list[Document] = splitter.split_documents(docs)

token_max: int = 1000


def length_function(documents: list[Document]) -> int:
    """Get number of tokens for input contents.

    Args:
        documents (list[Document]): list of documents.

    Returns:
        int: length in documents.

    """
    return sum(chat.get_num_tokens(doc.page_content) for doc in documents)


class OverallState(TypedDict):
    """Overall state."""

    contents: list[str]  # noqa: WPS
    summaries: Annotated[list, operator.add]
    collapsed_summaries: list[Document]
    final_summary: str


class SummaryState(TypedDict):
    """Content state."""

    content: str  # noqa: WPS


async def generate_summary(state: SummaryState) -> dict[str, list[Any]]:
    """Generate summary for document.

    Args:
        state (SummaryState): summary state.

    Returns:
        dict[str, list[Any]]: dict with response.

    """
    prompt: PromptValue = map_prompt.invoke(state["content"])
    response: BaseMessage = await chat.ainvoke(prompt)
    return {"summaries": [response.content]}


def map_summaries(state: OverallState) -> list[Send]:
    """Map summaries.

    Args:
        state (OverallState): overall state.

    Returns:
        list[Send]: list of send messages.

    """
    return [
        Send("generate_summary", {"content": content})
        for content in state["contents"]  # noqa: WPS
    ]


def collect_summaries(state: OverallState) -> dict[str, list[Document]]:
    """Collect summaries.

    Args:
        state (OverallState): overall states.

    Returns:
        dict[str, list[Document]]: list of documents.

    """
    return {
        "collapsed_summaries": [
            Document(summary) for summary in state["summaries"]
        ],
    }


async def _reduce(system_input: dict) -> str | list[dict[Any, Any], Any]:
    prompt: PromptValue = reduce_prompt.invoke(system_input)
    response: BaseMessage = await chat.ainvoke(prompt)
    return response.content


async def collapse_summaries(state: OverallState) -> dict[str, list[Document]]:
    """Collapse summaries.

    Args:
        state (OverallState): overall state.

    Returns:
        dict[str, list[Document]]: dict with response.

    """
    doc_lists: list[list[Document]] = split_list_of_docs(
        docs=state["collapsed_summaries"],
        length_func=length_function,
        token_max=token_max,
    )

    doc_list: list[Document] = [
        await acollapse_docs(doc_list, _reduce) for doc_list in doc_lists
    ]

    return {"collapsed_summaries": doc_list}


def should_collapse(
    state: OverallState,
) -> Literal["collapse_summaries", "generate_final_summary"]:
    """Check is should collapse.

    Returns:
        Literal["collapse_summaries", "generate_final_summary"]: one of these.

    """
    num_tokens: int = length_function(state["collapsed_summaries"])

    if num_tokens > token_max:
        return "collapse_summaries"

    return "generate_final_summary"


async def generate_final_summary(state: OverallState) -> dict[dict, Any]:
    """Generate final summary.

    Args:
        state (OverallState): overall state.

    Returns:
        dict[dict, Any]: dict with response.

    """
    response: Any = await _reduce(state["collapsed_summaries"])
    return {"final_summary": response}


graph: StateGraph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)
graph.add_node("collect_summaries", collect_summaries)
graph.add_node("collapse_summaries", collapse_summaries)
graph.add_node("generate_final_summary", generate_final_summary)

graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
graph.add_edge("generate_summary", "collect_summaries")
graph.add_conditional_edges("collect_summaries", should_collapse)
graph.add_conditional_edges("collapse_summaries", should_collapse)
graph.add_edge("generate_final_summary", END)

app: CompiledStateGraph = graph.compile()
