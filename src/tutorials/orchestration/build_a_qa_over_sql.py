"""Tutorial module.

Build a Question/Answering system over SQL data.

Notes:
* Looks like structured output can't handle SQL queries.

"""
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langchain import hub
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama, OllamaLLM
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.settings import llm_model_name, llm_model_temperature, llm_model_url

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
