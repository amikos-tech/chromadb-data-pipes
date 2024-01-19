from typing import (
    Optional,
    Sequence,
    Any,
    Dict,
    Union,
    Protocol,
    Generator,
    TypeVar,
    Iterable,
    Generic,
)

from chromadb.api.types import Embedding
from pydantic import BaseModel, Field

C = TypeVar("C")


class ResourceFeature(BaseModel, Generic[C]):
    feature_name: str
    feature_type: C


Metadata = Dict[str, Union[str, int, float, bool]]


class EmbeddableResource(BaseModel):
    id: Optional[str] = Field(None, description="Document ID")
    metadata: Optional[Metadata] = Field(None, description="Document metadata")
    embedding: Optional[Embedding] = Field(None, description="Document embedding")

    @staticmethod
    def resource_features() -> Sequence[ResourceFeature]:
        return [
            ResourceFeature[Embedding](
                feature_name="embedding", feature_type=Embedding
            ),
            ResourceFeature[Metadata](feature_name="metadata", eature_type=Metadata),
            ResourceFeature[str](feature_name="id", feature_type=str),
        ]


class EmbeddableTextResource(EmbeddableResource):
    text_chunk: Optional[str] = Field(None, description="Document text chunk")

    @staticmethod
    def resource_features() -> Sequence[ResourceFeature]:
        return [
            ResourceFeature[str](feature_name="text_chunk", feature_type=str),
            *super().resource_features(),
        ]


D = TypeVar("D", bound=EmbeddableResource, contravariant=True)


class ChromaDocumentSourceGenerator(Protocol[D]):
    def __iter__(self) -> Generator[D, None, None]:
        ...


class CdpProducer(Protocol[D]):
    def produce(
        self, limit: int = -1, offset: int = 0, **kwargs: Dict[str, Any]
    ) -> Iterable[D]:
        ...


class CdpProcessor(Protocol[D]):
    def process(self, *, documents: Iterable[D], **kwargs: Any) -> Iterable[D]:
        ...


class CdpConsumer(Protocol[D]):
    def consume(self, *, documents: Iterable[D], **kwargs: Dict[str, Any]) -> None:
        ...
