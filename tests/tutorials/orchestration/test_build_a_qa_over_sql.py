"""Test module.

Build a Question/Answering system over SQL data.
"""

from typing import Any

from src.tutorials.orchestration.build_a_qa_over_sql import (
    agent_executor,
    db,
    execute_query,
    graph,
    query_prompt_template,
    retriever_tool,
    system_message,
    write_query,
)


def test_db() -> None:
    """Test database."""
    tables: list[str] = [
        "Album",
        "Artist",
        "Customer",
        "Employee",
        "Genre",
        "Invoice",
        "InvoiceLine",
        "MediaType",
        "Playlist",
        "PlaylistTrack",
        "Track",
    ]

    db_query: str = "SELECT * FROM Artist LIMIT 10;"
    db_query_output: str = (
        "[(1, 'AC/DC'), (2, 'Accept'), (3, 'Aerosmith'), "
        "(4, 'Alanis Morissette'), (5, 'Alice In Chains'), "
        "(6, 'Antônio Carlos Jobim'), (7, 'Apocalyptica'), "
        "(8, 'Audioslave'), (9, 'BackBeat'), (10, 'Billy Cobham')]"
    )

    assert db.dialect == "sqlite"
    assert db.get_usable_table_names() == tables
    assert db.run(db_query) == db_query_output


def test_query_prompt() -> None:
    """Test query prompt content."""
    template: str = (
        "Given an input question, create a syntactically correct {dialect} "
        "query to run to help find the answer. "
        "Unless the user specifies in his question a specific number of "
        "examples they wish to obtain, "
        "always limit your query to at most {top_k} results. "
        "You can order the results by a relevant column to return the most "
        "interesting examples in the database.\n\n"
        "Never query for all the columns from a specific table, "
        "only ask for a the few relevant columns given the question.\n\n"
        "Pay attention to use only the column names that you can see in the "
        "schema description. "
        "Be careful to not query for columns that do not exist. "
        "Also, pay attention to which column is in which table.\n\n"
        "Only use the following tables:\n{table_info}\n\n"
        "Question: {input}"
    )

    assert len(query_prompt_template.messages) == 1
    assert query_prompt_template.messages[0].prompt.template == template


def test_write_query() -> None:
    """Test write query."""
    user_input: str = "How many Employees are there?"
    expected_output: str = "SELECT COUNT(*) FROM Employee"

    assert write_query({"question": user_input})["query"] == expected_output


def test_query_execute() -> None:
    """Test query execute."""
    ai_input: str = "SELECT COUNT(*) FROM Employee"
    expected_output: dict[str: Any] = {"result": "[(8,)]"}

    assert execute_query({"query": ai_input}) == expected_output


def test_graph() -> None:
    """Test graph."""
    user_input: str = "How many employees are there?"

    expected_output_1: dict[str, Any] = {
        "write_query": {"query": "SELECT COUNT(*) FROM Employee"},
    }

    expected_output_2: dict[str, Any] = {
        "execute_query": {"result": "[(8,)]"},
    }

    expected_output_3: dict[str, Any] = {
        "generate_answer": {"answer": "The number of employees is 8."},
    }

    response: Any = graph.invoke(
        input={"question": user_input},
        stream_mode="updates",
    )

    assert response[0] == expected_output_1
    assert response[1] == expected_output_2
    assert response[2] == expected_output_3


def test_agent_prompt() -> None:
    """Test query prompt content."""
    template: str = (
        "System: You are an agent designed to interact with a SQL database.\n"
        "Given an input question, create a syntactically correct SQLite "
        "query to run, then look at the results of the query and return the "
        "answer.\n"
        "Unless the user specifies a specific number of examples they wish to "
        "obtain, always limit your query to at most 5 results.\n"
        "You can order the results by a relevant column to return the most "
        "interesting examples in the database.\n"
        "Never query for all the columns from a specific table, "
        "only ask for the relevant columns given the question.\n"
        "You have access to tools for interacting with the database.\n"
        "Only use the below tools. Only use the information returned by the "
        "below tools to construct your final answer.\n"
        "You MUST double check your query before executing it. "
        "If you get an error while executing a query, "
        "rewrite the query and try again.\n\n"
        "DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) "
        "to the database.\n\n"
        "To start you should ALWAYS look at the tables in the database to "
        "see what you can query.\n"
        "Do NOT skip this step.\n"
        "Then you should query the schema of the most relevant tables."
    )

    assert system_message == template


# This one is very buggy
# def test_agent_executor() -> None:
#     """Test agent executor."""
#     user_input: str = "Which country's customers spent the most?"

#     response: Any = agent_executor.invoke(
#         input={"messages": [{"role": "user", "content": user_input}]},
#     )

#     assert "Japan" in response["messages"][-1].content


def test_retriever_tool() -> None:
    """Test retriever tool."""
    user_input: str = "Alice Chains"
    expected_output: str = (
        "Alice In Chains\n\n"
        "Somewhere in Time\n\n"
        "Stone Temple Pilots\n\n"
        "Velvet Revolver\n\n"
        "System Of A Down"
    )

    assert retriever_tool.invoke(user_input) == expected_output


# This one is very buggy
# def test_agent_executor() -> None:
#     """Test agent."""
#     user_input: str = "How many albums does alis in chain have?"

#     response: Any = agent_executor.invoke(
#         input={"messages": [{"role": "user", "content": user_input}]},
#     )

#     assert "1" in response["messages"][-1].content
