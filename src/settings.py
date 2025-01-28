"""Settings module."""

from typing import Final

llm_model_url: Final[str] = "http://localhost:11434"
llm_model_name: Final[str] = "llama3.2:3b"
llm_model_temperature: Final[float] = 0

embedding_model_url: Final[str] = "http://localhost:11435"
embedding_model_name: Final[str] = "bge-m3"
