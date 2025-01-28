"""Tutorial module.

Build a semantic search engine module.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.settings import embedding_model_name, embedding_model_url

documents: list[Document] = [
    Document(
        "Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        "Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

loader: PyPDFLoader = PyPDFLoader("data/nke-10k-2023.pdf")
docs: list[Document] = loader.load()

_chunk_size: int = 1000
_chunk_overlap: int = 200
text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
    chunk_size=_chunk_size, chunk_overlap=_chunk_overlap, add_start_index=True,
)

all_splits: list[Document] = text_splitter.split_documents(docs)

embeddings: OllamaEmbeddings = OllamaEmbeddings(
    base_url=embedding_model_url,
    model=embedding_model_name,
)

first_vector: list[float] = embeddings.embed_query(all_splits[0].page_content)
second_vector: list[float] = embeddings.embed_query(all_splits[1].page_content)

vector_store: Milvus = Milvus(
    embedding_function=embeddings,
    auto_id=True,
)

# Remove all data in milvus
vector_store.delete(expr="pk > 0")

ids: list[str] = vector_store.add_documents(all_splits)
