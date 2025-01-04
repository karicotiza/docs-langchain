"""Build a semantic search engine module."""

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


documents: list[Document] = [
    Document(
        'Dogs are great companions, known for their loyalty and friendliness.',
        metadata={'source': 'mammal-pets-doc'},
    ),
    Document(
        'Cats are independent pets that often enjoy their own space.',
        metadata={'source': 'mammal-pets-doc'},
    )
]

loader: PyPDFLoader = PyPDFLoader('data/nke-10k-2023.pdf')
docs: list[Document] = loader.load()
