import os
from urllib.parse import urlparse, parse_qs

import chromadb
import typer
from enum import Enum
from typing import Annotated, Optional, List

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from pydantic import BaseModel, Field

from chroma_dp.huggingface.hf_export import HFExportRequest, run as export_run
from chroma_dp.huggingface.hf_import import HFImportRequest, run as import_run
from chroma_dp.utils.chroma import CDPUri, get_client_for_uri


class SupportedEmbeddingFunctions(str, Enum):
    default = "default"


hf_commands = typer.Typer()


class HFImportUri(BaseModel):
    dataset: Optional[str] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    split: Optional[str] = None
    stream: Optional[bool] = None
    id_feature: Optional[str] = None
    doc_feature: Optional[str] = None
    embed_feature: Optional[str] = None
    meta_features: Optional[List[str]] = None
    private: Optional[bool] = Field(False, description="Make dataset private on Hugging Face Hub. "
                                                       "Note: This parameter is only applicable to exports.")

    @staticmethod
    def from_uri(uri: str) -> "HFImportUri":
        parsed_uri = urlparse(uri)
        query_params = parse_qs(parsed_uri.query)

        if parsed_uri.scheme != "hf":
            raise ValueError(f"Unsupported scheme: {parsed_uri.scheme}. Must be 'hf:`")
        dataset = parsed_uri.path
        limit = query_params.get("limit", [None])[0]
        offset = query_params.get("offset", [None])[0]
        split = query_params.get("split", [None])[0]
        stream = query_params.get("stream", [None])[0]
        id_feature = query_params.get("id_feature", [None])[0]
        doc_feature = query_params.get("doc_feature", [None])[0]
        embed_feature = query_params.get("embed_feature", [None])[0]
        meta_features = query_params.get("meta_features", [None])[0]
        private = query_params.get("private", [None])[0]

        return HFImportUri(
            dataset=dataset,
            limit=limit,
            offset=offset,
            split=split,
            stream=stream,
            id_feature=id_feature,
            doc_feature=doc_feature,
            embed_feature=embed_feature,
            meta_features=meta_features.split(",") if meta_features else None,
            private=private,
        )


@hf_commands.command(name="import", help="Import HF dataset into Chroma.", no_args_is_help=True)  # type: ignore
def hf_import(
        dataset: Annotated[
            Optional[str],
            typer.Option(
                help="The HuggingFace dataset. Expected format: <user>/<dataset_id>."
            ),
        ] = None,
        doc_feature: Annotated[str, typer.Option(help="The document feature.")] = None,
        chroma_endpoint: Annotated[str, typer.Option(help="The Chroma endpoint.")] = None,
        collection: Annotated[str, typer.Option(help="The Chroma collection.")] = None,
        split: Annotated[
            Optional[str], typer.Option(help="The Hugging Face")
        ] = "train",
        stream: Annotated[
            bool, typer.Option(help="Stream dataset instead of downloading.")
        ] = False,
        create: Annotated[
            bool, typer.Option(help="Create the Chroma collection if it does not exist.")
        ] = False,
        embed_feature: Annotated[
            str, typer.Option(help="The embedding feature.")
        ] = None,
        meta_features: Annotated[
            List[str], typer.Option(help="The metadata features.")
        ] = None,
        id_feature: Annotated[str, typer.Option(help="The id feature.")] = None,
        limit: Annotated[int, typer.Option(help="The limit.")] = -1,
        offset: Annotated[int, typer.Option(help="The offset.")] = 0,
        batch_size: Annotated[int, typer.Option(help="The batch size.")] = 100,
        embedding_function: Annotated[
            SupportedEmbeddingFunctions, typer.Option(help="The embedding function.")
        ] = SupportedEmbeddingFunctions.default,
        upsert: Annotated[bool, typer.Option(help="Upsert documents.")] = False,
        cdp_uri: Annotated[str, typer.Option(help="The ChromaDP URI.")] = None,
        hf_uri: Annotated[str, typer.Option(help="The Hugging Face URI.")] = None,
):
    if not embed_feature and embedding_function == SupportedEmbeddingFunctions.default:
        _embedding_function = ONNXMiniLM_L6_V2()
    if not embed_feature and embedding_function not in SupportedEmbeddingFunctions:
        raise ValueError(f"Unsupported embedding function: {embedding_function}")
    if cdp_uri is None and chroma_endpoint is None:
        raise ValueError("Please provide a ChromaDP URI or a Chroma endpoint.")
    _collection = collection
    _batch_size = batch_size
    _upsert = upsert
    _create = create
    _dataset = dataset
    _limit = limit
    _offset = offset
    _split = split
    _stream = stream
    _id_feature = id_feature
    _doc_feature = doc_feature
    _embed_feature = embed_feature
    _meta_features = meta_features
    _embedding_function = None

    if cdp_uri is not None:
        parsed_uri = CDPUri.from_uri(cdp_uri)
        client = get_client_for_uri(parsed_uri)
        _collection = parsed_uri.collection
        _batch_size = parsed_uri.batch_size or batch_size
        _offset = parsed_uri.offset or offset
        _upsert = parsed_uri.upsert or upsert
        _create = parsed_uri.create_collection or create
    else:
        client = chromadb.HttpClient(host=chroma_endpoint)

    if hf_uri is not None:
        _hf_uri = HFImportUri.from_uri(hf_uri)
        _dataset = _hf_uri.dataset
        _limit = _hf_uri.limit or limit
        _offset = _hf_uri.offset or offset
        _split = _hf_uri.split or split
        _stream = _hf_uri.stream or stream
        _id_feature = _hf_uri.id_feature or id_feature
        _doc_feature = _hf_uri.doc_feature or doc_feature
        _embed_feature = _hf_uri.embed_feature or embed_feature
        _meta_features = _hf_uri.meta_features or meta_features
    _import_request = HFImportRequest(
        client=client,
        collection=_collection,
        dataset=_dataset,
        create_collection=_create,
        limit=_limit,
        dataset_split=_split,
        dataset_stream=_stream,
        embedding_function=_embedding_function,
        document_feature=_doc_feature,
        id_feature=_id_feature,
        embedding_feature=_embed_feature,
        metadata_features=_meta_features,
        upsert=_upsert,
        offset=_offset,
        batch_size=_batch_size,

    )

    response = import_run(_import_request)
    typer.echo(response)


@hf_commands.command(name="export", help="Export Chroma collection to HF dataset.",
                     no_args_is_help=True)  # type: ignore
def hf_export(
        chroma_endpoint: Annotated[str, typer.Option(help="The Chroma endpoint.")] = None,
        collection: Annotated[str, typer.Option(help="The Chroma collection.")] = None,
        dataset: Annotated[str, typer.Option(help="The Hugging Face dataset.")] = None,
        split: Annotated[
            Optional[str], typer.Option(help="The Hugging Face")
        ] = "train",
        meta_features: Annotated[
            List[str], typer.Option(help="The metadata features.")
        ] = None,
        limit: Annotated[int, typer.Option(help="The limit.")] = -1,
        offset: Annotated[int, typer.Option(help="The offset.")] = 0,
        batch_size: Annotated[int, typer.Option(help="The batch size.")] = 100,
        upload: Annotated[bool, typer.Option(help="Upload")] = False,
        private: Annotated[bool, typer.Option(help="Private HF Repo")] = False,
        cdp_uri: Annotated[str, typer.Option(help="The ChromaDP URI.")] = None,
        hf_uri: Annotated[str, typer.Option(help="The Hugging Face URI.")] = None,
        out: Annotated[str, typer.Option(help="The output path.")] = None,
):
    if cdp_uri is None and chroma_endpoint is None:
        raise ValueError("Please provide a ChromaDP URI or a Chroma endpoint.")
    _collection = collection
    _batch_size = batch_size
    _dataset = dataset
    _limit = limit
    _offset = offset
    _split = split
    _upload = upload
    _private = private

    if cdp_uri is not None:
        parsed_uri = CDPUri.from_uri(cdp_uri)
        client = get_client_for_uri(parsed_uri)
        _collection = parsed_uri.collection
        _batch_size = parsed_uri.batch_size or batch_size
        _offset = parsed_uri.offset or offset
        _limit = parsed_uri.limit or limit
    else:
        client = chromadb.HttpClient(host=chroma_endpoint)

    if hf_uri is not None:
        _hf_uri = HFImportUri.from_uri(hf_uri)
        _dataset = _hf_uri.dataset
        _split = _hf_uri.split or split
        _meta_features = _hf_uri.meta_features or meta_features
        _private = _hf_uri.private if _hf_uri.private is not None else private  # do we need it this way?

    export_request = HFExportRequest(
        client=client,
        collection=_collection,
        dataset=_dataset,
        split=_split,
        output_path=out or os.path.join(os.getcwd(), _dataset.split("/")[-1]),
        upload=_upload,
        private=_private,
    )

    rest = export_run(export_request)
    typer.echo(rest)
