from inspect import signature
from typing import Optional, Sequence, Any, Dict, Union, Protocol, Generator, TypeVar, Iterable

from chromadb import EmbeddingFunction
from chromadb.api import ClientAPI
from chromadb.api.types import Embedding
from pydantic import BaseModel, Field, field_validator


class ChromaDocument(BaseModel):
    id: Optional[str] = Field(None, description="Document ID")
    text_chunk: Optional[str] = Field(None, description="Document text chunk")
    metadata: Optional[Dict[str, Union[str, int, float, bool]]] = Field(None, description="Document metadata")
    embedding: Optional[Embedding] = Field(None, description="Document embedding")


D = TypeVar("D", bound=ChromaDocument, contravariant=True)


class ChromaDocumentSourceGenerator(Protocol[D]):
    def __iter__(self) -> Generator[ChromaDocument, None, None]:
        ...


class CdpProducer(Protocol[D]):
    def produce(self, limit: int = -1, offset: int = 0, **kwargs: Dict[str, Any]) -> Iterable[D]:
        ...


class CdpFilter(Protocol[D]):
    def filter(self, *, documents: Iterable[D], **kwargs: Dict[str, Any]) -> Iterable[D]:
        ...


class CdpTransformer(Protocol[D]):
    def transform(self, *, documents: Iterable[D], **kwargs: Dict[str, Any]) -> Iterable[D]:
        ...


class CdpConsumer(Protocol[D]):
    def consume(self, *, documents: Iterable[D], **kwargs: Dict[str, Any]) -> None:
        ...
