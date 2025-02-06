"""Tutorial module.

Build a Question/Answering system over SQL data.

Notes:
* Looks like structured output can't handle SQL queries.

"""
import ast
import re
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langchain import hub
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.tools.simple import Tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledGraph, CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from src.settings import (
    embedding_model_name,
    embedding_model_url,
    llm_model_name,
    llm_model_temperature,
    llm_model_url,
)

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import Runnable

db: SQLDatabase = SQLDatabase.from_uri("sqlite:///data/Chinook.db")


class State(TypedDict):
    """State structure."""

    question: str
    query: str
    result: str  # noqa: WPS
    answer: str


chat: ChatOllama = ChatOllama(
    base_url=llm_model_url,
    model=llm_model_name,
    temperature=llm_model_temperature,
)

llm: OllamaLLM = OllamaLLM(
    base_url=llm_model_url,
    model=llm_model_name,
    temperature=llm_model_temperature,
)

query_prompt_template: Any = hub.pull("langchain-ai/sql-query-system-prompt")


# class QueryOutput(TypedDict):
#     """Generated SQL query."""

#     query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: State) -> dict[str, Any]:
    """Generate SQL query to fetch information.

    Args:
        state (State): state structure.

    Returns:
        dict[str, Any]: dict with response.

    """
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        },
    )

    # structured_llm: Runnable = chat.with_structured_output(QueryOutput)
    # result: Any = structured_llm.invoke(prompt)  # noqa: WPS
    # return {"query": result["query"]}
    result: Any = llm.invoke(prompt)  # noqa: WPS
    return {"query": result}


def execute_query(state: State) -> dict[str, Any]:
    """Execute SQL query.

    Args:
        state (State): state structure.

    Returns:
        dict[str, Any]: dict with response.

    """
    execute_query_tool: QuerySQLDatabaseTool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}


def generate_answer(state: State) -> dict[str, Any]:
    """Answer question using retrieved information as context.

    Args:
        state (State): state structure.

    Returns:
        dict[str, Any]: dict with response.

    """
    prompt: str = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response: BaseMessage = chat.invoke(prompt)

    return {"answer": response.content}


graph_builder: StateGraph = StateGraph(State)
graph_builder.add_sequence([write_query, execute_query, generate_answer])
graph_builder.add_edge(START, "write_query")
graph: CompiledStateGraph = graph_builder.compile()

toolkit: SQLDatabaseToolkit = SQLDatabaseToolkit(db=db, llm=chat)
tools: list[BaseTool] = toolkit.get_tools()

prompt_template: Any = hub.pull("langchain-ai/sql-agent-system-prompt")
system_message = prompt_template.format(dialect="SQLite", top_k=5)

agent_executor: CompiledGraph = create_react_agent(
    model=chat,
    tools=tools,
    prompt=system_message,
)


def _query_as_list(db: SQLDatabase, query: str) -> list[str]:
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))


artists: list[str] = _query_as_list(db, "SELECT Name FROM Artist")
albums: list[str] = _query_as_list(db, "SELECT Title FROM Album")

embeddings: OllamaEmbeddings = OllamaEmbeddings(
    base_url=embedding_model_url,
    model=embedding_model_name,
)

vector_store: InMemoryVectorStore = InMemoryVectorStore(embeddings)
ids: list[str] = vector_store.add_texts(artists + albums)
retriever: VectorStoreRetriever = vector_store.as_retriever(
    search_kwargs={"k": 5},
)

description: str = (
    "Use to look up values to filter on. Input is an approximate spelling "
    "of the proper noun, output is valid proper nouns. Use the noun most "
    "similar to the search."
)

retriever_tool: Tool = create_retriever_tool(
    retriever,
    name="search_proper_nouns",
    description=description,
)

suffix: str = (
    "If you need to filter on a proper noun like a Name, "
    "you must ALWAYS first look up the filter value using the "
    "'search_proper_nouns' tool! Do not try to guess at the proper name - "
    "use this function to find similar ones."
)

system = f"{system_message}\n\n{suffix}"
tools.append(retriever_tool)
agent: CompiledGraph = create_react_agent(chat, tools, prompt=system)
