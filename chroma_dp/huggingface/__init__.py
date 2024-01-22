import json

import sys
from typing import Annotated, Optional, List, Generator, Union, Sequence, Any, Dict
from urllib.parse import urlparse, parse_qs

import datasets
import typer
from datasets import load_dataset, Dataset, concatenate_datasets
from datasets.config import HF_ENDPOINT
from huggingface_hub import DatasetCard, HfApi
from pydantic import BaseModel, Field

from chroma_dp import ChromaDocumentSourceGenerator, EmbeddableTextResource
from chroma_dp.huggingface.utils import _infer_hf_type, int_or_none, bool_or_false
from chroma_dp.utils.chroma import remap_features

hf_commands = typer.Typer()


class HFImportRequest(BaseModel):
    dataset: Union[str, Dataset]
    split: Optional[str] = Field("train", description="The Hugging Face dataset split")
    stream: Optional[bool] = Field(
        False, description="Stream dataset instead of downloading."
    )
    limit: Optional[int] = None
    offset: Optional[int] = None
    document_feature: Optional[str] = Field(..., description="Document feature")
    id_feature: Optional[str] = Field(None, description="ID feature")
    embedding_feature: Optional[str] = Field(None, description="Embedding feature")
    metadata_features: Optional[Sequence[str]] = Field(
        None, description="Metadata features"
    )
    batch_size: Optional[int] = Field(100, description="Batch size")

    class Config:
        arbitrary_types_allowed = True


def _doc_wrapper(
    row: Any,
    document_feature: str,
    embedding_feature: Optional[str],
    id_feature: Optional[str],
    metadata_features: Optional[Sequence[str]],
) -> EmbeddableTextResource:
    doc = EmbeddableTextResource(
        id=row[id_feature] if id_feature else None,
        text_chunk=row[document_feature],
        metadata={k: row[k] for k in metadata_features} if metadata_features else None,
        embedding=row[embedding_feature] if embedding_feature else None,
    )
    return doc


class HFChromaDocumentSourceGenerator(
    ChromaDocumentSourceGenerator[EmbeddableTextResource]
):
    """
    A generator of chroma document from a Hugging Face dataset.
    """

    def __init__(self, import_request: HFImportRequest):
        if isinstance(import_request.dataset, str):
            self._dataset = load_dataset(
                import_request.dataset,
                split=import_request.split,
                streaming=import_request.stream,
            )
        else:
            self._dataset = import_request.dataset

        if (
            not import_request.document_feature
            or import_request.document_feature not in self._dataset.features.keys()
        ):
            raise ValueError(
                f"Document column {import_request.document_feature} not found in dataset"
            )
        self._batch_size = import_request.batch_size or 100
        self._doc_feature = import_request.document_feature
        self._id_feature = import_request.id_feature
        self._embed_feature = import_request.embedding_feature
        self._meta_features = import_request.metadata_features
        self._extract_features = [self._doc_feature]
        if self._id_feature:
            self._extract_features.append(self._id_feature)
        if self._embed_feature:
            self._extract_features.append(self._embed_feature)
        if import_request.metadata_features:
            if not all(
                _metadata_column in self._dataset.features
                for _metadata_column in import_request.metadata_features
            ):
                missing_features = [
                    f
                    for f in import_request.metadata_features
                    if f not in self._dataset.features
                ]
                raise ValueError(
                    f"Metadata feature(s) {missing_features} not found in dataset features {self._dataset.features.keys()}"
                )
            self._extract_features.extend(import_request.metadata_features)

        _dataset_len = (
            self._dataset.num_rows if hasattr(self._dataset, "num_rows") else -1
        )
        if _dataset_len > 0:
            self._limit = (
                min(import_request.limit, _dataset_len)
                if import_request.limit != -1
                else _dataset_len
            )
        else:
            self._limit = import_request.limit or -1
        self._offset = import_request.offset or 0
        self._stream = import_request.stream or False

    def _get_batch(self, offset: int, limit: int) -> Dataset:
        return self._dataset[offset : offset + limit]

    def __iter__(self) -> Generator[EmbeddableTextResource, None, None]:
        if self._stream:
            yield from self._streaming_iterator()
        else:
            end = self._offset + self._limit
            for start in range(self._offset, end, self._batch_size):
                subset = self._dataset[start : min(start + self._batch_size, end)]
                yield from [
                    _doc_wrapper(
                        dict(zip(self._extract_features, values)),
                        self._doc_feature,
                        self._embed_feature,
                        self._id_feature,
                        self._meta_features,
                    )
                    for values in zip(*(subset[key] for key in self._extract_features))
                ]

    def _streaming_iterator(self) -> Generator[EmbeddableTextResource, None, None]:
        count = 0
        for item in self._dataset:
            if count < self._offset:
                continue
            if self._limit is not None and 0 < self._limit <= count:
                break

            yield item
            count += 1


class HFImportUri(BaseModel):
    dataset: Optional[str] = None
    dataset_name: Optional[str] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    split: Optional[str] = None
    stream: Optional[bool] = None
    id_feature: Optional[str] = None
    doc_feature: Optional[str] = None
    embed_feature: Optional[str] = None
    is_remote: Optional[bool] = None
    meta_features: Optional[List[str]] = None
    private: Optional[bool] = Field(
        False,
        description="Make dataset private on Hugging Face Hub. "
        "Note: This parameter is only applicable to exports.",
    )
    batch_size: Optional[int] = Field(100, description="Batch size")

    @staticmethod
    def from_uri(uri: str) -> "HFImportUri":
        parsed_uri = urlparse(uri)
        query_params = parse_qs(parsed_uri.query)

        if parsed_uri.scheme not in ["file", "hf"]:
            raise ValueError(
                f"Unsupported scheme: {parsed_uri.scheme}. Must be 'hf:` or `file:`."
            )
        dataset = (parsed_uri.hostname or "") + parsed_uri.path
        is_remote = False
        if parsed_uri.scheme == "hf":
            is_remote = True
        dataset_name = parsed_uri.path
        limit = int_or_none(query_params.get("limit", [None])[0])
        offset = int_or_none(query_params.get("offset", [None])[0])
        split = query_params.get("split", [None])[0]
        stream = bool_or_false(query_params.get("stream", [False])[0])
        id_feature = query_params.get("id_feature", [None])[0]
        doc_feature = query_params.get("doc_feature", [None])[0]
        embed_feature = query_params.get("embed_feature", [None])[0]
        meta_features = query_params.get("meta_features", [None])[0]
        private = bool_or_false(query_params.get("private", [False])[0])
        batch_size = int_or_none(query_params.get("batch_size", [100])[0])

        return HFImportUri(
            dataset=dataset,
            dataset_name=dataset_name,
            limit=limit,
            offset=offset,
            split=split,
            stream=stream,
            id_feature=id_feature,
            doc_feature=doc_feature,
            embed_feature=embed_feature,
            meta_features=meta_features.split(",") if meta_features else None,
            private=private,
            is_remote=is_remote,
            batch_size=batch_size,
        )


def hf_import(
    uri: Annotated[
        str, typer.Argument(help="Dataset uri. eg. `hf://user/dataset?split=train`")
    ],
    split: Annotated[
        Optional[str], typer.Option(help="The HuggingFace dataset split")
    ] = "train",
    stream: Annotated[
        bool, typer.Option(help="Stream dataset instead of downloading.")
    ] = False,
    doc_feature: Annotated[
        str, typer.Option(help="The document feature.")
    ] = "document",
    embed_feature: Annotated[
        Optional[str], typer.Option(help="The embedding feature.")
    ] = "embedding",
    meta_features: Annotated[
        Optional[List[str]], typer.Option(help="The metadata features.")
    ] = None,
    id_feature: Annotated[Optional[str], typer.Option(help="The id feature.")] = "id",
    limit: Annotated[Optional[int], typer.Option(help="The limit.")] = -1,
    offset: Annotated[Optional[int], typer.Option(help="The offset.")] = 0,
    batch_size: Optional[int] = typer.Option(
        100, "--batch-size", "-b", help="The batch size."
    ),
) -> None:
    _hf_uri = HFImportUri.from_uri(uri)
    _dataset = _hf_uri.dataset
    _limit = _hf_uri.limit or limit
    _offset = _hf_uri.offset or offset
    _split = _hf_uri.split or split
    _stream = _hf_uri.stream or stream
    _id_feature = _hf_uri.id_feature or id_feature
    _doc_feature = _hf_uri.doc_feature or doc_feature
    _embed_feature = _hf_uri.embed_feature or embed_feature
    _meta_features = _hf_uri.meta_features or meta_features
    _batch_size = batch_size or _hf_uri.batch_size

    import_request = HFImportRequest(
        dataset=_dataset,
        split=_split,
        stream=_stream,
        limit=_limit,
        offset=_offset,
        document_feature=_doc_feature,
        id_feature=_id_feature,
        embedding_feature=_embed_feature,
        metadata_features=_meta_features,
        batch_size=_batch_size,
    )
    gen = HFChromaDocumentSourceGenerator(import_request)
    for doc in gen:
        typer.echo(json.dumps(doc.model_dump()))


def hf_export(
    uri: Annotated[
        str,
        typer.Argument(
            help="Dataset uri. eg. `hf:user/dataset?split=train` or `file:dataset-name?split=train`"
        ),
    ],
    inf: typer.FileText = typer.Argument(sys.stdin),
    split: Annotated[
        Optional[str], typer.Option(help="The HuggingFace dataset split")
    ] = "train",
    doc_feature: Annotated[
        str, typer.Option(help="The document feature.")
    ] = "text_chunk",
    embed_feature: Annotated[
        Optional[str], typer.Option(help="The embedding feature.")
    ] = "embedding",
    meta_features: Annotated[
        Optional[List[str]], typer.Option(help="The metadata features.")
    ] = None,
    id_feature: Annotated[str, typer.Option(help="The id feature.")] = "id",
    limit: Annotated[int, typer.Option(help="The limit.")] = -1,
    offset: Annotated[int, typer.Option(help="The offset.")] = 0,
    batch_size: Annotated[int, typer.Option(help="The batch size.")] = 100,
    private: Annotated[
        bool, typer.Option(help="Make dataset private on Hugging Face Hub. ")
    ] = False,
) -> None:
    _hf_uri = HFImportUri.from_uri(uri)
    _dataset = _hf_uri.dataset
    _limit = _hf_uri.limit or limit
    _offset = _hf_uri.offset or offset
    _split = _hf_uri.split or split
    _id_feature = _hf_uri.id_feature or id_feature
    _doc_feature = _hf_uri.doc_feature or doc_feature
    _embed_feature = _hf_uri.embed_feature or embed_feature
    _meta_features = _hf_uri.meta_features or meta_features
    _batch_size = batch_size
    _private = _hf_uri.private or private
    _batch: Dict[str, Any] = {
        "id": [],
        "document": [],
        "embedding": [],
    }

    features = datasets.Features(
        {
            "id": datasets.Value("string"),
            "embedding": datasets.features.Sequence(
                feature=datasets.Value(dtype="float32")
            ),
            "document": datasets.Value("string"),
            # **(metadata_feature if metadata_feature else {})
        }
    )
    features.update()
    dataset = None
    for line in inf:
        doc = remap_features(
            json.loads(line),
            doc_feature,
            embed_feature=embed_feature,
            meta_features=meta_features,
            id_feature=id_feature,
        )

        _batch["id"].append(doc.id)
        _batch["document"].append(doc.text_chunk)
        _batch["embedding"].append(doc.embedding)
        if doc.metadata:
            for key in doc.metadata.keys():
                if f"metadata.{key}" not in features:
                    features[f"metadata.{key}"] = _infer_hf_type(doc.metadata[key])
                _batch[f"metadata.{key}"].append(doc.metadata[key])

        if len(_batch["document"]) >= _batch_size:
            if dataset is None:
                dataset = Dataset.from_dict(
                    _batch,
                    features=features,
                    info=datasets.DatasetInfo(
                        description="Chroma Collection export.", features=features
                    ),
                    split=_split,
                )
            else:
                new_dataset = Dataset.from_dict(
                    _batch,
                    features=features,
                    info=datasets.DatasetInfo(
                        description="Chroma Collection export.", features=features
                    ),
                    split=_split,
                )
                dataset = concatenate_datasets([dataset, new_dataset])
            _batch: Dict[str, Any] = {
                "id": [],
                "document": [],
                "embedding": [],
            }

    if len(_batch["document"]) > 0:
        if dataset is None:
            dataset = Dataset.from_dict(
                _batch,
                features=features,
                info=datasets.DatasetInfo(
                    description="Chroma Collection export.", features=features
                ),
                split=_split,
            )
        else:
            new_dataset = Dataset.from_dict(
                _batch,
                features=features,
                info=datasets.DatasetInfo(
                    description="Chroma Collection export.", features=features
                ),
                split=_split,
            )
            dataset = concatenate_datasets([dataset, new_dataset])
    dataset.save_to_disk("test_dataset")

    if _hf_uri.is_remote:
        dataset.push_to_hub(_hf_uri.dataset, private=_private)
        custom_metadata = {
            "license": "mit",
            "language": "en",
            "pretty_name": f"Chroma export of collection N/A",
            "size_categories": ["n<1K"],
            "x-chroma": {
                "description": "Chroma Dataset",
                "collection": "N/A",
                "metadata": "N/A",
            },
        }
        card = DatasetCard.load(repo_id_or_path=_hf_uri.dataset, repo_type="dataset")
        data_info = card.data
        data_dict = {**data_info.to_dict(), **custom_metadata}
        card.content = f"---\n{str(data_dict)}\n---\n{card.text}"
        HfApi(endpoint=HF_ENDPOINT).upload_file(
            path_or_fileobj=str(card).encode(),
            path_in_repo="README.md",
            repo_id=_hf_uri.dataset,
            repo_type="dataset",
        )
