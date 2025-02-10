"""Tutorial module.

Build a Question Answering application over a Graph Database.
"""

from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_ollama import ChatOllama

from src.settings import (
    llm_model_name,
    llm_model_temperature,
    llm_model_url,
    neo4j_password,
    neo4j_url,
    neo4j_user,
    phi4_model_name,
    phi4_model_temperature,
    phi4_model_url,
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

llama_chat: ChatOllama = ChatOllama(
    base_url=llm_model_url,
    model=llm_model_name,
    temperature=llm_model_temperature,
)

phi4_chat: ChatOllama = ChatOllama(
    base_url=phi4_model_url,
    model=phi4_model_name,
    temperature=phi4_model_temperature,
)

llama_chain: GraphCypherQAChain = GraphCypherQAChain.from_llm(
    graph=enhanced_graph,
    llm=llama_chat,
    verbose=True,
    allow_dangerous_requests=True,
)

phi4_chain: GraphCypherQAChain = GraphCypherQAChain.from_llm(
    graph=enhanced_graph,
    llm=phi4_chat,
    verbose=True,
    allow_dangerous_requests=True,
)
