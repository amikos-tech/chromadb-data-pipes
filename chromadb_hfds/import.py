import uuid
from typing import Optional, Union, Sequence, TypedDict
import os
import chromadb
from chromadb import EmbeddingFunction
from chromadb.api import ServerAPI
from chromadb.api.models import Collection
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from datasets import Dataset, load_dataset
from rich.progress import track


class ImportRequest(TypedDict):
    dataset: Union[str, Dataset]
    dataset_split: str
    dataset_stream: bool
    client: ServerAPI
    endpoint: str
    collection: str
    create: bool
    document_column: str
    embedding_column: str
    metadata_columns: Sequence[str]
    id_column: str
    limit: int
    offset: int
    batch_size: int
    embedding_function: EmbeddingFunction


class ImportResult(TypedDict):
    pass


class ChromaDBImportBuilder(object):

    def __init__(self, import_request: Optional[ImportRequest] = None) -> None:
        self._dataset = import_request.get('dataset') if import_request else None
        self._chroma_endpoint = import_request.get('endpoint') if import_request else None
        self._collection = import_request.get('collection') if import_request else None
        self._create = import_request.get('create') if import_request else None
        self._dataset_split = import_request.get('dataset_split') if import_request else 'train'
        self._dataset_stream = import_request.get('dataset_stream') if import_request else False
        self._document_column = import_request.get('document_column') if import_request else None
        self._embedding_column = import_request.get('embedding_column') if import_request else None
        self._metadata_columns = import_request.get('metadata_columns') if import_request else None
        self._id_column = import_request.get('id_column') if import_request else None
        self._limit = import_request.get('limit') if import_request else -1
        self._offset = import_request.get('offset') if import_request else None
        self._batch_size = import_request.get('batch_size') if import_request else 100
        self._embedding_function = import_request.get('embedding_function') \
            if import_request else DefaultEmbeddingFunction()
        self._chroma_client = chromadb.HttpClient(host=self._chroma_endpoint)

    @classmethod
    def from_dataset(cls, dataset: Union[str, Dataset], split: str = 'train',
                     stream: bool = False) -> "ChromaDBImportBuilder":
        _instance = cls()
        _instance._dataset = dataset
        _instance._dataset_split = split
        _instance._dataset_stream = stream
        return _instance

    def into(self, endpoint: str, collection: str, create: bool = True) -> "ChromaDBImportBuilder":
        self._chroma_endpoint = endpoint or self._chroma_endpoint
        self._collection = collection or self._collection
        self._create = create
        if "CHROMA_TOKEN" in os.environ:
            # TODO: Add token auth
            pass
        self._chroma_client = chromadb.HttpClient(host=self._chroma_endpoint)
        return self

    def into_chroma(self, chroma_client: ServerAPI, collection: str, create: bool = True) -> "ChromaDBImportBuilder":
        self._chroma_client = chroma_client
        self._collection = collection or self._collection
        self._create = create
        return self

    def with_document_column(self, document_column: str) -> "ChromaDBImportBuilder":
        self._document_column = document_column
        return self

    def with_embedding_column(self, embedding_column: str) -> "ChromaDBImportBuilder":
        self._embedding_column = embedding_column
        return self

    def with_metadata_columns(self, metadata_columns: Sequence[str]) -> "ChromaDBImportBuilder":
        self._metadata_columns = metadata_columns
        return self

    def with_id_column(self, id_column: str) -> "ChromaDBImportBuilder":
        self._id_column = id_column
        return self

    def with_range(self, limit: Optional[int], offset: Optional[int]) -> "ChromaDBImportBuilder":
        self._limit = limit
        self._offset = offset
        return self

    def with_batch_size(self, batch_size: int) -> "ChromaDBImportBuilder":
        self._batch_size = batch_size
        return self

    def with_embedding_function(self, embedding_function: EmbeddingFunction) -> "ChromaDBImportBuilder":
        self._embedding_function = embedding_function
        return self

    def run(self) -> ImportResult:

        # TODO: validate params
        # TODO test connectivity to Chroma
        self._chroma_client.heartbeat()
        try:
            collection = self._chroma_client.get_collection(self._collection)
        except Exception as e:
            if self._create:
                collection = self._chroma_client.create_collection(self._collection)
            else:
                raise e
        if self._dataset is None:
            raise ValueError("Dataset not provided")
        if isinstance(self._dataset, str):
            _dataset = load_dataset(
                self._dataset, split=self._dataset_split, streaming=self._dataset_stream)
        else:
            _dataset = self._dataset
        _dataset_len = _dataset.num_rows if hasattr(
            _dataset, "num_rows") else -1
        _limit = min(self._limit, _dataset_len) if self._limit != -1 else _dataset_len
        # validate dataset params
        features = _dataset.features
        if self._document_column not in features.keys():
            raise ValueError(
                f"Document column {self._document_column} not found in dataset")
        _should_embed = True
        if self._embedding_column is not None and self._embedding_column not in features.keys():
            raise ValueError(
                f"Embedding column {self._embedding_column} not found in dataset")
        if self._embedding_column is not None and self._embedding_column in features.keys():
            _should_embed = False
            collection._embedding_function = self._embedding_function  # TODO check if embedding function defined?
        if self._metadata_columns is not None:
            for _metadata_column in self._metadata_columns:
                if _metadata_column not in features.keys():
                    raise ValueError(
                        f"Metadata column {_metadata_column} not found in dataset")
        _should_generate_ids = True
        if self._id_column is not None and self._id_column not in features.keys():
            raise ValueError(
                f"ID column {self._id_column} not found in dataset")
        if self._id_column is not None and self._id_column in features.keys():
            _should_generate_ids = False

        _batch = {
            "documents": [],
            "embeddings": [],
            "metadatas": [],
            "ids": [],
        }

        def add_to_col(col: Collection, batch: dict):
            try:
                col.add(**batch)
            except Exception as e:
                print(e)

        for i in track(range(0, _limit, self._batch_size), description="Importing Dataset"):
            ds_batch = _dataset[i:min(
                i + self._batch_size, _limit)]
            if len(ds_batch[self._document_column]) == 0:
                break
            _batch["documents"].extend(ds_batch[self._document_column])
            if not _should_embed:
                _batch["embeddings"].extend(
                    ds_batch[self._embedding_column])
            else:
                del _batch["embeddings"]
            if not _should_generate_ids:
                _batch["ids"].extend(ds_batch[self._id_column])
            else:
                _batch["ids"].extend([str(uuid.uuid4()) for _ in range(len(ds_batch[self._document_column]))])
            if len(self._metadata_columns) > 0:
                _batch["metadatas"].extend([dict(zip(self._metadata_columns, values)) for values in zip(
                    *[ds_batch[feature] for feature in self._metadata_columns])])
            else:
                _batch["metadatas"].append(None)
            # executor.submit(add_to_col, _batch.copy())
            add_to_col(collection, _batch.copy())
            _batch = {
                "documents": [],
                "embeddings": [],
                "metadatas": [],
                "ids": [],
            }

        return ImportResult()


if __name__ == "__main__":
    import_request = ImportRequest(
        dataset="KShivendu/dbpedia-entities-openai-1M",
        dataset_split="train",
        dataset_stream=False,
        endpoint="http://localhost:8000",
        collection="dbpedia",
        create=True,
        document_column="text",
        embedding_column="openai",
        metadata_columns=["title"],
        id_column="_id",
        limit=10000,
        offset=0,
        batch_size=100,
        embedding_function=DefaultEmbeddingFunction()
    )

    ChromaDBImportBuilder(import_request).run()
