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


class ImportRequest(BaseModel):
    client: Optional[ClientAPI] = Field(None, description="ChromaDB client")
    collection: Optional[str] = Field(None, description="ChromaDB collection")
    collection_metadata: Optional[dict] = Field(None,
                                                description="Collection metadata. "
                                                            "Such as hnsw:batch_size and hnsw:sync_threshold."
                                                            "Only works if create_collection is True.")
    create_collection: Optional[bool] = Field(True, description="Create collection if it does not exist")
    document_feature: Optional[str] = Field(..., description="Document feature")
    embedding_feature: Optional[str] = Field(None, description="Embedding feature")
    metadata_features: Sequence[str] = Field([], description="Metadata features")
    id_feature: Optional[str] = Field(None, description="ID feature")
    limit: Optional[int] = Field(None, description="Limit")
    offset: Optional[int] = Field(0, description="Offset")
    batch_size: Optional[int] = Field(100, description="Batch size")
    embedding_function: Optional[Any] = Field(None, description="Embedding function")
    upsert: Optional[bool] = Field(False, description="Upsert documents, such as matched by their Id.")

    @field_validator('embedding_function')
    def check_embedding_function(cls, embedding_function):
        # Here you can add your validation logic
        # For example, check if v is an instance of your EmbeddingFunction protocol
        function_signature = signature(
            embedding_function.__class__.__call__
        ).parameters.keys()
        protocol_signature = signature(EmbeddingFunction.__call__).parameters.keys()

        if embedding_function and not function_signature == protocol_signature:
            raise ValueError(
                f"Embedding function {embedding_function} does not implement the EmbeddingFunction protocol"
            )
        return embedding_function

    class Config:
        arbitrary_types_allowed = True


class ImportResult(BaseModel):
    success: bool = Field(..., description="Whether the import was successful")


class ExportRequest(BaseModel):
    client: ClientAPI = Field(..., description="ChromaDB client")
    collection: Optional[str] = Field(..., description="ChromaDB collection")
    limit: Optional[int] = Field(-1, description="Limit")
    offset: Optional[int] = Field(0, description="Offset")
    batch_size: Optional[int] = Field(100, description="Batch size")
    embedding_feature: Optional[str] = Field(None, description="Name of the embedding feature in the dataset")
    metadata_features: Optional[Union[Sequence[str], Dict[str, str]]] = Field([],
                                                                              description="Names or mapping of metadata features")

    class Config:
        arbitrary_types_allowed = True


class ExportResult(BaseModel):
    success: bool = Field(..., description="Whether the export was successful")
    message: Optional[str] = Field(None, description="Message")
