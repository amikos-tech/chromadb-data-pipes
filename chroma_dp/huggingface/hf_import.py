import uuid
from typing import Union, Optional, Generic, TypeVar, Type, Dict, Any

from chromadb.api.models import Collection
from datasets import Dataset, load_dataset, Features
from pydantic import Field, BaseModel
from rich.progress import track

from chroma_dp import ImportRequest, ImportResult


class HFImportRequest(ImportRequest):
    dataset: Union[str, Dataset]
    dataset_split: str
    dataset_stream: Optional[bool] = Field(False, description="Stream dataset instead of downloading.")


TModel = TypeVar('TModel', bound=BaseModel)


class HFImportRequestBuilder(Generic[TModel]):
    model: Type[TModel]
    values: Dict[str, object]

    def __init__(self, model: Type[TModel]) -> None:
        super().__setattr__('model', model)
        super().__setattr__('values', {})

    def __setattr__(self, name: str, value: object) -> None:
        self.values[name] = value

    def build(self):
        return self.model(**self.values)


class _HFImportRequest(BaseModel):
    collection: Collection
    dataset: Dataset
    limit: int
    offset: Optional[int] = None
    dataset_features: Optional[Features] = None
    should_embed: Optional[bool] = True
    should_generate_ids: Optional[bool] = True

    class Config:
        arbitrary_types_allowed = True


def validate(import_request: HFImportRequest) -> _HFImportRequest:
    _ir_builder = HFImportRequestBuilder(_HFImportRequest)
    _client = import_request.client
    _client.heartbeat()
    # TODO: validate version compatibility
    try:
        _ir_builder.collection = _client.get_collection(import_request.collection)
    except Exception as e:
        if import_request.create_collection:
            _ir_builder.collection = _client.create_collection(import_request.collection,
                                                               metadata=import_request.collection_metadata)
        else:
            raise e
    if isinstance(import_request.dataset, str):
        _dataset = load_dataset(
            import_request.dataset, split=import_request.dataset_split, streaming=import_request.dataset_stream
        )
    else:
        _dataset = import_request.dataset
    _ir_builder.dataset = _dataset
    _dataset_len = _dataset.num_rows if hasattr(_dataset, "num_rows") else -1
    _limit = min(import_request.limit, _dataset_len) if import_request.limit != -1 else _dataset_len
    _ir_builder.limit = _limit
    _ir_builder.offset = import_request.offset
    _features = _dataset.features
    _ir_builder.dataset_features = _features
    if import_request.document_feature not in _features.keys():
        raise ValueError(
            f"Document column {import_request.document_feature} not found in dataset"
        )
    _should_embed = True
    if (
            import_request.embedding_feature is not None
            and import_request.embedding_feature not in _features.keys()
    ):
        raise ValueError(
            f"Embedding feature {import_request.embedding_feature} not found in dataset features {_features.keys()}"
        )
    if (
            import_request.embedding_feature is not None
            and import_request.embedding_feature in _features.keys()
    ):
        _should_embed = False
    else:
        _ir_builder.collection = _client.get_collection(import_request.collection,
                                                        embedding_function=import_request.embedding_function)
    _ir_builder.should_embed = _should_embed

    if import_request.metadata_features is not None and len(import_request.metadata_features) > 0:
        for _metadata_column in import_request.metadata_features:
            if _metadata_column not in _features.keys():
                raise ValueError(
                    f"Metadata feature {_metadata_column} not found in dataset features {_features.keys()}"
                )
    _should_generate_ids = True
    if import_request.id_feature is not None and import_request.id_feature not in _features.keys():
        raise ValueError(f"ID column {import_request.id_feature} not found in dataset features {_features.keys()}")
    if import_request.id_feature is not None and import_request.id_feature in _features.keys():
        _should_generate_ids = False
    _ir_builder.should_generate_ids = _should_generate_ids
    return _ir_builder.build()


# TODO move this to a common implementation
def run(import_request: HFImportRequest) -> ImportResult:
    # validate dataset params
    _ir = validate(import_request)
    _chroma_client = import_request.client
    _batch: Dict[str, Any] = {
        "documents": [],
        "embeddings": [],
        "metadatas": [],
        "ids": [],
    }

    def add_to_col(col: Collection, batch: Dict[str, Any]) -> None:
        try:
            if import_request.upsert:
                col.upsert(**batch)
            else:
                col.add(**batch)
        except Exception as e:
            raise e

    for i in track(
            range(0, _ir.limit, import_request.batch_size), description="Importing Dataset"
    ):
        ds_batch = _ir.dataset[i: min(i + import_request.batch_size, _ir.limit)]
        if len(ds_batch[import_request.document_feature]) == 0:
            break
        _batch["documents"].extend(ds_batch[import_request.document_feature])
        if not _ir.should_embed:
            _batch["embeddings"].extend(ds_batch[import_request.embedding_feature])
        else:
            del _batch["embeddings"]
        if not _ir.should_generate_ids:
            _batch["ids"].extend(ds_batch[import_request.id_feature])
        else:
            # TODO: make ID generation a pluggable strategy
            _batch["ids"].extend(
                [
                    str(uuid.uuid4())
                    for _ in range(len(ds_batch[import_request.document_feature]))
                ]
            )
        if len(import_request.metadata_features) > 0:
            _batch["metadatas"].extend(
                [
                    dict(zip(import_request.metadata_features, values))
                    for values in zip(
                    *[ds_batch[feature] for feature in import_request.metadata_features]
                )
                ]
            )
        else:
            del _batch["metadatas"]
        # executor.submit(add_to_col, _batch.copy())
        add_to_col(_ir.collection, _batch.copy())
        _batch = {
            "documents": [],
            "embeddings": [],
            "metadatas": [],
            "ids": [],
        }

    return ImportResult(success=True, message="Import successful.")
