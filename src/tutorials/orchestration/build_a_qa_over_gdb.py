"""Tutorial module.

Build a Question Answering application over a Graph Database.

Notes:
* LLaMa3.2:3B is't capable with Cypher, so it can't pass the
"test_graph_cypher_qa_chain" test.
* Phi4 is capable with Cypher, but doesn't support function calling. So it can
pass the "test_graph_cypher_qa_chain" test, but can't pass the "test_flow"
test. I will leave commented code here and in "./build/prod.yml" to run phi4.
* Right now my machine can't launch LLMs as big as LLaMa3.3:70B, but i guess
it can pas both tests.

"""
from __future__ import annotations
from enum import StrEnum
from dataclasses import dataclass


from operator import add
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph, Neo4jVector
from langchain_neo4j.chains.graph_qa.cypher_utils import (
    CypherQueryCorrector,
    Schema,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from neo4j.exceptions import CypherSyntaxError
from pydantic import BaseModel, Field

from src.settings import (
    embedding_model_name,
    embedding_model_url,
    llm_model_name,
    llm_model_temperature,
    llm_model_url,
    neo4j_password,
    neo4j_url,
    neo4j_user,
    # phi4_model_name,
    # phi4_model_temperature,
    # phi4_model_url,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableSerializable


class Role(StrEnum):
    """LangChain role adapter."""

    system = "system"
    human = "human"


@dataclass(slots=True)
class Message:
    """Message structure."""

    role: Role
    text: str


    def items(self) -> tuple[str, str]:  # noqa: WPS
        """Get structure as key, value.

        Returns:
            tuple[str, str]: role as a key and text as a value.

        """
        return (self.role, self.text)


def prompt(*args: Message) -> ChatPromptTemplate:
    """Make prompt for langchain.

    Args:
        args (list[Message]): list of tuple with roles and
            messages. You can set placeholders for values like "{value}", they
            will be invoked by model or chain later.

    Returns:
        ChatPromptTemplate: prepared ChatPromptTemplate.

    """
    return ChatPromptTemplate.from_messages(
        messages=[message.items() for message in args],
    )


graph: Neo4jGraph = Neo4jGraph(
    url=neo4j_url,
    username=neo4j_user,
    password=neo4j_password,
)

drop_query: str = "MATCH (n) DETACH DELETE n"

graph.query(drop_query)

movies_query: str = (
    "LOAD CSV WITH HEADERS FROM "
    "'https://raw.githubusercontent.com/tomasonjo/blog-datasets/"
    "main/movies/movies_small.csv' "
    "AS row "

    "MERGE (m:Movie {id:row.movieId}) "

    "SET m.released = date(row.released),"
    "    m.title = row.title,"
    "    m.imdbRating = toFloat(row.imdbRating) "

    "FOREACH (director in split(row.director, '|') |"
    "    MERGE (p:Person {name:trim(director)})"
    "    MERGE (p)-[:DIRECTED]->(m)) "

    "FOREACH (actor in split(row.actors, '|') |"
    "    MERGE (p:Person {name:trim(actor)})"
    "    MERGE (p)-[:ACTED_IN]->(m)) "

    "FOREACH (genre in split(row.genres, '|') |"
    "    MERGE (g:Genre {name:trim(genre)})"
    "    MERGE (m)-[:IN_GENRE]->(g))"
)

graph.query(movies_query)

enhanced_graph: Neo4jGraph = Neo4jGraph(
    url=neo4j_url,
    username=neo4j_user,
    password=neo4j_password,
    enhanced_schema=True,
)

chat: ChatOllama = ChatOllama(
    base_url=llm_model_url,
    model=llm_model_name,
    temperature=llm_model_temperature,
)

chain: GraphCypherQAChain = GraphCypherQAChain.from_llm(
    graph=enhanced_graph,
    llm=chat,
    verbose=True,
    allow_dangerous_requests=True,
)

# phi4_chat: ChatOllama = ChatOllama(
#     base_url=phi4_model_url,
#     model=phi4_model_name,
#     temperature=phi4_model_temperature,
# )

# phi4_chain: GraphCypherQAChain = GraphCypherQAChain.from_llm(
#     graph=enhanced_graph,
#     llm=phi4_chat,
#     verbose=True,
#     allow_dangerous_requests=True,
# )


class InputState(TypedDict):
    """Input state."""

    question: str


class OverallState(TypedDict):
    """Overall state."""

    question: str
    next_action: str
    cypher_statement: str
    cypher_errors: list[str]
    database_records: list[dict]
    steps: Annotated[list[str], add]


class OutputState(TypedDict):
    """Output state."""

    answer: str
    steps: list[str]
    cypher_statement: str


prompt_template: str = (
    "As an intelligent assistant, your primary objective is to decide "
    "whether a given question is related to movies or not."
    'If the question is related to movies, output "movie". '
    'Otherwise, output "end". '
    "To make this decision, assess the content of the question "
    "and determine if it refers to any movie, actor, director, film industry, "
    'or related topics. Provide only the specified output: "movie" or "end".'
    "\n\nQuestion: {question}"
)

guardrails_prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
    prompt_template,
)


class GuardrailsOutput(BaseModel):
    """Guard rails output."""

    decision: Literal["movie", "end"] = Field(
        description="Decision on whether the question is related to movies",
    )


guardrails_chain: RunnableSerializable[dict, dict | BaseModel] = (
    guardrails_prompt
    | chat.with_structured_output(GuardrailsOutput)
)


def guardrails(state: InputState) -> OverallState:
    """Decide if the question is related to movies or not.

    Args:
        state (InputState): input state.

    Returns:
        OverallState: overall state.

    """
    guardrails_output: dict | BaseModel = guardrails_chain.invoke(
        input={"question": state.get("question")},
    )

    database_records: str | None = None

    if guardrails_output.decision == "end":
        database_records = (
            "This questions is not about movies or their cast. "
            "Therefore I cannot answer this question."
        )

    return {
        "next_action": guardrails_output.decision,
        "database_records": database_records,
        "steps": ["guardrail"],
    }


examples: list[dict[str, str]] = [
    {
        "question": "How many artists are there?",
        "query": (
            "MATCH (a:Person)-[:ACTED_IN]->(:Movie) RETURN count(DISTINCT a)"
        ),
    },
    {
        "question": "Which actors played in the movie Casino?",
        "query": (
            "MATCH (m:Movie {title: 'Casino'})<-[:ACTED_IN]-(a) RETURN a.name"
        ),
    },
    {
        "question": "How many movies has Tom Hanks acted in?",
        "query": (
            "MATCH (a:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) "
            "RETURN count(m)"
        ),
    },
    {
        "question": "List all the genres of the movie Schindler's List",
        "query": (
            "MATCH (m:Movie {title: 'Schindler's List'})-"
            "[:IN_GENRE]->(g:Genre) RETURN g.name"
        ),
    },
    {
        "question": (
            "Which actors have worked in movies from both "
            "the comedy and action genres?"
        ),
        "query": (
            "MATCH (a:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genre), "
            "(a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genre) "
            "WHERE g1.name = 'Comedy' AND g2.name = 'Action' "
            "RETURN DISTINCT a.name"
        ),
    },
    {
        "question": (
            "Which directors have made movies with at least "
            "three different actors named 'John'?"
        ),
        "query": (
            "MATCH (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Person) "
            "WHERE a.name STARTS WITH 'John' "
            "WITH d, COUNT(DISTINCT a) AS JohnsCount "
            "WHERE JohnsCount >= 3 RETURN d.name"
        ),
    },
    {
        "question": (
            "Identify movies where directors also played a role in the film."
        ),
        "query": (
            "MATCH (p:Person)-[:DIRECTED]->(m:Movie), (p)-[:ACTED_IN]->(m) "
            "RETURN m.title, p.name"
        ),
    },
    {
        "question": (
            "Find the actor with the highest number of movies in the database."
        ),
        "query": (
            "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) "
            "RETURN a.name, "
            "COUNT(m) AS movieCount ORDER BY movieCount DESC LIMIT 1"
        ),
    },
]


embeddings: OllamaEmbeddings = OllamaEmbeddings(
    base_url=embedding_model_url,
    model=embedding_model_name,
)


example_selector: SemanticSimilarityExampleSelector = (
    SemanticSimilarityExampleSelector.from_examples(
        examples=examples,
        embeddings=embeddings,
        vectorstore_cls=Neo4jVector,
        k=5,
        input_keys=["question"],
        url=neo4j_url,
        username=neo4j_user,
        password=neo4j_password,
    )
)

text2cypher_template: str = (
    "Given an input question, convert it to a Cypher query. No pre-amble."
    "Do not wrap the response in any backticks or anything else. "
    "Respond with a Cypher statement only!\n\n"

    "You are a Neo4j expert. Given an input question, "
    "create a syntactically correct Cypher query to run. "
    "Do not wrap the response in any backticks or anything else. "
    "Respond with a Cypher statement only! Here is the schema information\n\n"
    "{schema}\n\n"

    "Below are a number of examples of questions "
    "and their corresponding Cypher queries.\n\n"
    "{fewshot_examples}\n\n"

    "User input: {question}\n\n"
    "Cypher query: "
)

text2cypher_prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
    text2cypher_template,
)

text2cypher_chain: RunnableSerializable[dict, str] = (
    text2cypher_prompt
    | chain
    | StrOutputParser()
)


def generate_cypher(state: OverallState) -> OverallState:
    """Generate a cypher statement based on the provided schema and user input.

    Args:
        state (OverallState): overall state.

    Returns:
        OverallState: overall state.

    """
    fewshot_examples: list[str] = [
        f"Question: {element['question']}\nCypher:{element['query']}"
        for element
        in example_selector.select_examples(
            input_variables={"question": state["question"]},
        )
    ]

    try:
        generated_cypher = text2cypher_chain.invoke(
            {
                "question": state.get("question"),
                "fewshot_examples": "\n\n".join(fewshot_examples),
                "schema": enhanced_graph.schema,
            },
        )
    except CypherSyntaxError:
        return {
            "cypher_statement": "",
            "steps": ["generate_cypher"],
        }

    return {
        "cypher_statement": generated_cypher,
        "steps": ["generate_cypher"],
    }


validate_cypher_template: str = (
    "You are a Cypher expert "
    "reviewing a statement written by a junior developer.\n\n"

    "You must check the following:\n"
    "* Are there any syntax errors in the Cypher statement?\n"
    "* Are there any missing or undefined variables in the Cypher statement?\n"
    "* Are any node labels missing from the schema?\n"
    "* Are any relationship types missing from the schema?\n"
    "* Are any of the properties not included in the schema?\n"
    "* Does the Cypher statement include enough information "
    "to answer the question?\n\n"

    "Examples of good errors:\n"
    "* Label (:Foo) does not exist, did you mean (:Bar)?\n"
    "* Property bar does not exist for label Foo, did you mean baz?\n"
    "* Relationship FOO does not exist, did you mean FOO_BAR?\n\n"

    "Schema:\n{schema}\n\n"
    "The question is:\n{question}\n\n"
    "The Cypher statement is:\n{cypher}\n\n"
    "Make sure you don't make any mistakes!"
)

validate_cypher_prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
    validate_cypher_template,
)


class Property(BaseModel):
    """Property structure."""

    node_label: str = Field(
        description="The label of the node to which this property belongs.",
    )

    property_key: str = Field(
        description="The key of the property being filtered.",
    )

    property_value: str = Field(
        description="The value that the property is being matched against.",
    )


class ValidateCypherOutput(BaseModel):
    """Validate Cypher output structure."""

    errors: list[str] | None = Field(
        description=(
            "A list of syntax or semantical errors in the Cypher statement. "
            "Always explain the discrepancy between schema "
            "and Cypher statement"
        ),
    )

    filters: list[Property] | None = Field(
        description=(
            "A list of property-based filters applied in the Cypher statement."
        ),
    )


validate_cypher_chain: RunnableSerializable[dict, dict | BaseModel] = (
    validate_cypher_prompt
    | chat.with_structured_output(ValidateCypherOutput)
)

corrector_schema: list[Schema] = [
    Schema(element["start"], element["type"], element["end"])
    for element
    in enhanced_graph.structured_schema["relationships"]
]

cypher_query_corrector: CypherQueryCorrector = CypherQueryCorrector(
    schemas=corrector_schema,
)


def _check_filter(llm_filter: Any) -> bool:
    return next(
        prop
        for prop
        in enhanced_graph.structured_schema["node_props"][filter.node_label]
        if prop["property"] == filter.property_key
    )["type"] != "STRING"


def _next_action(mapping_errors: list, errors: list) -> str:
    if mapping_errors:
        return "end"

    if errors:
        return "correct_cypher"

    return "execute_cypher"


def validate_cypher(state: OverallState) -> OverallState:
    """Validate Cypher statements and maps any property values to the database.

    Args:
        state (OverallState): overall state.

    Returns:
        OverallState: overall state.

    """
    errors: list = []
    mapping_errors: list = []

    try:
        enhanced_graph.query(f"EXPLAIN {state['cypher_statement']}")
    except CypherSyntaxError as cypher_syntax_error:
        errors.append(cypher_syntax_error.message)

    corrected_cypher: str = cypher_query_corrector(state["cypher_statement"])

    if not corrected_cypher:
        errors.append("The generated Cypher statement doesn't fit the schema")

    llm_output: dict | BaseModel = validate_cypher_chain.invoke(
        input={
            "question": state["question"],
            "schema": enhanced_graph.schema,
            "cypher": state["cypher_statement"],
        },
    )

    if llm_output.errors:
        errors.extend(llm_output.errors)

    if llm_output.filters:
        for llm_filter in llm_output.filters:
            if _check_filter(llm_filter):
                continue

            mapping = enhanced_graph.query(
                f"MATCH (n:{filter.node_label}) WHERE toLower(n.`{filter.property_key}`) = toLower($value) RETURN 'yes' LIMIT 1",  # noqa: E501
                {"value": filter.property_value},
            )

            if not mapping:
                mapping_errors.append(
                    f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"  # noqa: E501
                )

    next_action: str = _next_action(mapping_errors, errors)

    return {
        "next_action": next_action,
        "cypher_statement": corrected_cypher,
        "cypher_errors": errors,
        "steps": ["validate_cypher"],
    }


correct_cypher_template: str = (
    "You are a Cypher expert reviewing a statement "
    "written by a junior developer. "
    "You need to correct the Cypher statement based on the provided errors. "
    "No pre-amble. Do not wrap the response in any backticks "
    "or anything else. Respond with a Cypher statement only!\n\n"
    "Check for invalid syntax or semantics "
    "and return a corrected Cypher statement.\n\n"
    "Schema:\n{schema}\n\n"
    "Note: Do not include any explanations or apologies in your responses. "
    "Do not wrap the response in any backticks or anything else. "
    "Respond with a Cypher statement only!\n\n"
    "Do not respond to any questions that might ask anything "
    "else than for you to construct a Cypher statement.\n\n"
    "The question is:\n{question}\n\n"
    "The Cypher statement is:\n{cypher}\n\n"
    "The errors are:\n{errors}\n\n"
    "Corrected Cypher statement: "
)


correct_cypher_prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
    template=correct_cypher_template,
)

correct_cypher_chain: RunnableSerializable[dict, str] = (
    correct_cypher_prompt
    | chat
    | StrOutputParser()
)


def correct_cypher(state: OverallState) -> OverallState:
    """Correct the Cypher statement based on the provided errors.

    Args:
        state (OverallState): overall state.

    Returns:
        OverallState: overall state.

    """
    corrected_cypher = correct_cypher_chain.invoke(
        input={
            "question": state.get("question"),
            "errors": state.get("cypher_errors"),
            "cypher": state.get("cypher_statement"),
            "schema": enhanced_graph.schema,
        },
    )

    return {
        "next_action": "validate_cypher",
        "cypher_statement": corrected_cypher,
        "steps": ["correct_cypher"],
    }


no_results: str = "I couldn't find any relevant information in the database"


def execute_cypher(state: OverallState) -> OverallState:
    """Execute the given Cypher statement.

    Args:
        state (OverallState): overall state.

    Returns:
        OverallState: overall state.

    """
    records: list[dict[str, Any]] = enhanced_graph.query(
        query=state["cypher_statement"],
    )

    return {
        "database_records": records if records else no_results,
        "next_action": "end",
        "steps": ["execute_cypher"],
    }


generate_final_prompt: ChatPromptTemplate = prompt(
    Message(
        role=Role.system,
        text="You are a helpful assistant",
    ),
    Message(
        role=Role.human,
        text=(
            "Use the following results retrieved from a database to "
            "provide\n a succinct, definitive answer to the user's "
            "question.\n\n"
            "Respond as if you are answering the question directly.\n\n"
            "Results: {results}\n"
            "Question: {question}"
        ),
    ),
)

generate_final_chain: RunnableSerializable[dict, str] = (
    generate_final_prompt
    | chat
    | StrOutputParser()
)


def generate_final_answer(state: OverallState) -> OutputState:
    """Decide if the question is related to movies.

    Args:
        state (OverallState): overall state.

    Returns:
        OutputState: output state.

    """
    final_answer = generate_final_chain.invoke(
        input={
            "question": state["question"],
            "results": state["database_records"],
        },
    )

    return {
        "answer": final_answer,
        "steps": ["generate_final_answer"],
    }


def guardrails_condition(
    state: OverallState,
) -> Literal["generate_cypher", "generate_final_answer"]:
    """Check guardrails condition.

    Returns:
        Literal["generate_cypher", "generate_final_answer"]: one of these

    """
    if state["next_action"] == "end":
        return "generate_final_answer"

    if state["next_action"] == "movie":
        return "generate_cypher"

    return "generate_cypher"


def validate_cypher_condition(
    state: OverallState,
) -> Literal["generate_final_answer", "correct_cypher", "execute_cypher"]:
    """Check guardrails condition.

    Returns:
        Literal["generate_final_answer", "correct_cypher", "execute_cypher"]:
            one of these.

    """
    if state.get("next_action") == "end":
        return "generate_final_answer"

    if state.get("next_action") == "correct_cypher":
        return "correct_cypher"

    if state.get("next_action") == "execute_cypher":
        return "execute_cypher"

    return "execute_cypher"


builder: StateGraph = StateGraph(
    state_schema=OverallState,
    input=InputState,
    output=OutputState,
)

builder.add_node(guardrails)
builder.add_node(generate_cypher)
builder.add_node(validate_cypher)
builder.add_node(correct_cypher)
builder.add_node(execute_cypher)
builder.add_node(generate_final_answer)

builder.add_edge(START, "guardrails")
builder.add_conditional_edges("guardrails", guardrails_condition)
builder.add_edge("generate_cypher", "validate_cypher")
builder.add_conditional_edges("validate_cypher", validate_cypher_condition)
builder.add_edge("execute_cypher", "generate_final_answer")
builder.add_edge("correct_cypher", "validate_cypher")
builder.add_edge("generate_final_answer", END)

flow: CompiledStateGraph = builder.compile()
