"""Test module.

Build a Question/Answering system over SQL data.
"""

from typing import Any

from src.tutorials.orchestration.build_a_qa_over_sql import (
    db,
    execute_query,
    graph,
    query_prompt_template,
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
