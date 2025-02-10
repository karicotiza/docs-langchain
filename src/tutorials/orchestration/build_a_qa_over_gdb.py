"""Tutorial module.

Build a Question Answering application over a Graph Database.
"""

from langchain_neo4j import Neo4jGraph

from src.settings import neo4j_password, neo4j_url, neo4j_user

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
