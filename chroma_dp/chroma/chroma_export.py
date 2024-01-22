import json
from typing import Annotated, Optional, List, Dict, Any
import typer
from chromadb import GetResult, Where, WhereDocument
from chromadb.api.models import Collection
from chromadb.api.types import validate_where, validate_where_document

from chroma_dp import EmbeddableTextResource
from chroma_dp.utils.chroma import CDPUri, get_client_for_uri


def _get_result_to_chroma_doc_list(result: GetResult) -> List[EmbeddableTextResource]:
    """Converts a GetResult to a list of ChromaDocuments."""
    docs = []
    for idx, _ in enumerate(result["ids"]):
        docs.append(
            EmbeddableTextResource(
                text_chunk=result["documents"][idx],
                embedding=result["embeddings"][idx],
                metadata=result["metadatas"][idx],
                id=result["ids"][idx],
            )
        )
    return docs


def remap_features(
    doc: EmbeddableTextResource,
    doc_feature: Optional[str] = "text_chunk",
    embed_feature: Optional[str] = "embedding",
    id_feature: str = "id",
    meta_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Remaps EmbeddableTextResource features to a dictionary."""

    _metas = (
        doc.metadata
        if meta_features is None
        else {
            k: doc.metadata[k]
            for k in meta_features
            if doc.metadata is not None and k in doc.metadata
        }
    )
    return {
        f"{doc_feature}": doc.text_chunk,
        f"{embed_feature}": doc.embedding,
        f"{id_feature}": doc.id,
        **(_metas if _metas is not None else {}),
    }


def read_large_data_in_chunks(
    collection: Collection,
    offset: int = 0,
    limit: int = 100,
    where: Where = None,
    where_document: WhereDocument = None,
) -> GetResult:
    """Reads large data in chunks from ChromaDB."""
    result = collection.get(
        where=where,
        where_document=where_document,
        limit=limit,
        offset=offset,
        include=["embeddings", "documents", "metadatas"],
    )
    return result


def chroma_export(
    uri: Annotated[str, typer.Argument(help="The Chroma endpoint.")],
    collection: Annotated[
        Optional[str], typer.Option(help="The Chroma collection.")
    ] = None,
    export_file: Optional[str] = typer.Option(
        None, "--out", help="Export .jsonl file."
    ),
    append: Annotated[bool, typer.Option(help="Append to export file.")] = False,
    limit: Annotated[int, typer.Option(help="The limit.")] = -1,
    offset: Annotated[int, typer.Option(help="The offset.")] = 0,
    batch_size: Annotated[int, typer.Option(help="The batch size.")] = 100,
    embed_feature: Annotated[
        str, typer.Option(help="The embedding feature.")
    ] = "embedding",
    meta_features: Annotated[
        Optional[List[str]], typer.Option(help="The metadata features.")
    ] = None,
    id_feature: Annotated[str, typer.Option(help="The id feature.")] = "id",
    doc_feature: Annotated[
        str, typer.Option(help="The document feature.")
    ] = "text_chunk",
    where: Optional[str] = typer.Option(
        None,
        "--where",
        "-m",
        help='Metadata filter. JSON with Chroma syntax is expected - \'{"metadata_key": "metadata_value"}\'',
    ),
    where_document: Optional[str] = typer.Option(
        None,
        "--where-document",
        "-d",
        help='Document filter string - JSON with Chroma syntax is expected - \'{"$contains": "search for this"}\'',
    ),
) -> None:
    if uri is None:
        raise ValueError("Please provide a ChromaDP URI.")
    parsed_uri = CDPUri.from_uri(uri)
    client = get_client_for_uri(parsed_uri)
    _collection = parsed_uri.collection or collection
    _batch_size = parsed_uri.batch_size or batch_size
    _offset = parsed_uri.offset or offset
    _limit = parsed_uri.limit or limit
    _start = _offset if _offset > 0 else 0
    chroma_collection = client.get_collection(_collection)
    col_count = chroma_collection.count()
    total_results_to_fetch = min(col_count, _limit) if _limit > 0 else col_count
    _where = None
    if where:
        _where = validate_where(json.loads(where))
    _where_document = None
    if where_document:
        _where_document = validate_where_document(json.loads(where_document))
    if export_file and not append:
        with open(export_file, "w") as f:
            f.write("")
    for offset in range(_start, total_results_to_fetch, _batch_size):
        _results = _get_result_to_chroma_doc_list(
            read_large_data_in_chunks(
                chroma_collection,
                offset=offset,
                limit=min(total_results_to_fetch - offset, _batch_size),
                where=_where,
                where_document=_where_document,
            )
        )
        _final_results = [
            remap_features(
                doc,
                doc_feature=doc_feature,
                embed_feature=embed_feature,
                id_feature=id_feature,
                meta_features=meta_features,
            )
            for doc in _results
        ]
        if export_file:
            with open(export_file, "a") as f:
                for _doc in _final_results:
                    f.write(json.dumps(_doc) + "\n")
        else:
            for _doc in _final_results:
                typer.echo(json.dumps(_doc))
