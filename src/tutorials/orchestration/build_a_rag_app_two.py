"""Tutorial module.

Build a Retrieval Augmented Generation (RAG) app: part 2.
"""
from __future__ import annotations

from contextlib import suppress
from typing import Any

from bs4 import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document  # noqa: TC002
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain_milvus import Milvus
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.state import CompiledStateGraph

from src.settings import (
    embedding_model_name,
    embedding_model_url,
    llm_model_name,
    llm_model_temperature,
    llm_model_url,
)

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
# with suppress(AttributeError):
#     vector_store.delete(expr="pk > 0")

# url: str = "https://lilianweng.github.io/posts/2023-06-23-agent/"
# loader: WebBaseLoader = WebBaseLoader(
#     web_path=url,
#     bs_kwargs={
#         "parse_only": SoupStrainer(
#             class_=("post-content", "post-title", "post-header"),
#         ),
#     },
# )

# docs: list[Document] = loader.load()

# chunk_size: int = 1000
# chuck_overlap: int = 200
# text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
#     chunk_size=chunk_size,
#     chunk_overlap=chuck_overlap,
#     add_start_index=True,
# )

# all_splits: list[Document] = text_splitter.split_documents(docs)

# for doc in all_splits:
#     doc.metadata["page"] = 0

# ids: list[str] = vector_store.add_documents(all_splits)

graph_builder: StateGraph = StateGraph(MessagesState)


@tool(response_format="content_and_artifact")
def retrieve(query: str) -> tuple[str, list[Document]]:
    """Retrieve information related to user query.

    Args:
        query (str): user query.

    Returns:
        tuple[str, list[Document]]: response as a string and a list with
        documents.

    """
    memory: list[str] = []
    retrieved_docs: list[Document] = vector_store.similarity_search(query, k=2)

    for doc in retrieved_docs:
        msg: str = f"Source: {doc.metadata}\nContent: {doc.page_content}"
        memory.append(msg)

    serialized: str = "\n\n".join(memory)

    return serialized, retrieved_docs


def query_or_respond(state: MessagesState) -> dict[str, list[Any]]:
    """Generate tool call for retrieval or respond.

    Args:
        state (MessagesState): messages state.

    Returns:
        dict[str, list[Any]]: dict with response.

    """
    llm_with_tools: Runnable = chat.bind_tools([retrieve])
    response: Any = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


tools: ToolNode = ToolNode([retrieve])


def _get_tool_messages(state: MessagesState) -> list[AnyMessage]:
    memory: list[AnyMessage] = []

    for message in reversed(state["messages"]):
        if message.type == "tool":
            memory.append(message)
        else:
            break

    return memory[::-1]


def _get_system_prompt_template() -> SystemMessagePromptTemplate:
    return SystemMessagePromptTemplate.from_template(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{docs_content}",
    )


def _get_conversation_messages(state: MessagesState) -> list[AnyMessage]:
    return [
        message for message in state["messages"] if (
            message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        )
    ]


def generate(state: MessagesState) -> dict[str, list[BaseMessage]]:
    """Generate response.

    Args:
        state (MessagesState): messages state.

    Returns:
        dict[str, list[BaseMessage]]: dict with response messages.

    """
    tool_messages: list[AnyMessage] = _get_tool_messages(state)
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_prompt: SystemMessage = SystemMessage(
        _get_system_prompt_template()
        .format(docs_content=docs_content)
        .content,
    )

    conversation_messages: list[AnyMessage] = _get_conversation_messages(state)
    response: BaseMessage = chat.invoke(
        input=[system_prompt, *conversation_messages],
    )

    return {"messages": [response]}


graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph: CompiledStateGraph = graph_builder.compile()
