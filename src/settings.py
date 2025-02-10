"""Settings module."""

from typing import Final

llm_model_url: Final[str] = "http://localhost:11434"
llm_model_name: Final[str] = "llama3.2:3b"
llm_model_temperature: Final[float] = 0

phi4_model_url: Final[str] = "http://localhost:11436"
phi4_model_name: Final[str] = "phi4"
phi4_model_temperature: Final[float] = 0

embedding_model_url: Final[str] = "http://localhost:11435"
embedding_model_name: Final[str] = "bge-m3"

neo4j_url: str = "bolt://localhost:7687"
neo4j_user: str = "neo4j"
neo4j_password: str = "neo4j"  # noqa: S105
