import numpy as np

try:
    import chromadb  # noqa: F401
except ImportError:
    raise ValueError(
        "The chromadb is not installed. This package (chromadbx) requires that Chroma is installed to work. "
        "Please install it with `pip install chromadb`"
    )
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
from pydantic import BaseModel, Field, ConfigDict

C = TypeVar("C")


class ResourceFeature(BaseModel, Generic[C]):
    feature_name: str
    feature_type: C


Metadata = Dict[str, Union[str, int, float, bool]]

EmbeddingWrapper = Union[Embedding, np.ndarray]


class EmbeddableResource(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: Optional[str] = Field(None, description="Document ID")
    metadata: Optional[Metadata] = Field(None, description="Document metadata")
    embedding: Optional[EmbeddingWrapper] = Field(
        None, description="Document embedding"
    )

    @staticmethod
    def resource_features() -> Sequence[ResourceFeature]:
        return [
            ResourceFeature[EmbeddingWrapper](
                feature_name="embedding", feature_type=EmbeddingWrapper
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

    def model_dump(self, **kwargs):
        # Convert NumPy arrays to lists before dumping
        data = super().model_dump(**kwargs)
        if isinstance(data["embedding"], np.ndarray):
            data["embedding"] = data["embedding"].tolist()
        return data


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
