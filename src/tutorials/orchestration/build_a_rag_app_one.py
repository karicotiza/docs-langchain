"""Tutorial module.

Build a Retrieval Augmented Generation (RAG) app: part 1.
"""
from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any, TypedDict

from bs4 import SoupStrainer
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document  # noqa: TC002
from langchain_milvus import Milvus
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph

from src.settings import (
    embedding_model_name,
    embedding_model_url,
    llm_model_name,
    llm_model_temperature,
    llm_model_url,
)

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.prompt_values import PromptValue
    from langgraph.graph.state import CompiledStateGraph

chat: ChatOllama = ChatOllama(
    base_url=llm_model_url,
    model=llm_model_name,
    temperature=llm_model_temperature,
)

embeddings: OllamaEmbeddings = OllamaEmbeddings(
    base_url=embedding_model_url,
    model=embedding_model_name,
)

vector_store: Milvus = Milvus(
    embedding_function=embeddings,
    auto_id=True,
)

# Remove all data in milvus
with suppress(AttributeError):
    vector_store.delete(expr="pk > 0")

url: str = "https://lilianweng.github.io/posts/2023-06-23-agent/"
loader: WebBaseLoader = WebBaseLoader(
    web_path=url,
    bs_kwargs={
        "parse_only": SoupStrainer(
            class_=("post-content", "post-title", "post-header"),
        ),
    },
)

docs: list[Document] = loader.load()

chunk_size: int = 1000
chuck_overlap: int = 200
text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chuck_overlap,
)

all_splits: list[Document] = text_splitter.split_documents(docs)

for doc in all_splits:
    doc.metadata["page"] = 0
    doc.metadata["start_index"] = 0

ids: list[str] = vector_store.add_documents(all_splits)

prompt_url: str = "rlm/rag-prompt"
prompt: Any = hub.pull(prompt_url)


class RagState(TypedDict):
    """RAG app state."""

    question: str
    context: list[Document]
    answer: str


def retrieve(state: RagState) -> dict[str, list[Document]]:
    """Retrieve documents.

    Args:
        state (RagState): rag state.

    Returns:
        dict[str, list[Document]]: dict with most similar documents.

    """
    retrieved_docs: list[Document] = vector_store.similarity_search(
        query=state["question"],
    )

    return {"context": retrieved_docs}


def _response_to_str(response: BaseMessage) -> str:
    answer: str | list[str | dict[Any, Any]] = response.content

    if isinstance(answer, list):
        return "".join(str(answer))

    return answer


def generate(state: RagState) -> dict[str, str]:
    """Generate answer.

    Args:
        state (RagState): rag state.

    Returns:
        dict[str, str]: dict with answer.

    """
    docs: str = "\n\n".join(doc.page_content for doc in state["context"])

    messages: PromptValue = prompt.invoke(
        input={
            "question": state["question"],
            "context": docs,
        },
    )

    response: BaseMessage = chat.invoke(messages)

    return {"answer": _response_to_str(response)}


graph_builder: StateGraph = StateGraph(RagState)
graph_builder.add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph: CompiledStateGraph = graph_builder.compile()
