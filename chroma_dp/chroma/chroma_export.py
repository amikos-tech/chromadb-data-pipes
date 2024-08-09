from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import orjson as json
from typing import Annotated, Optional, List, Dict, Any, Generator
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
        if meta_features is None or len(meta_features) == 0
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
    queue: Queue,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    where: Where = None,
    where_document: WhereDocument = None,
):
    """Reads large data in chunks from ChromaDB."""
    result = collection.get(
        where=where,
        where_document=where_document,
        limit=limit,
        offset=offset,
        include=["embeddings", "documents", "metadatas"],
    )
    try:
        queue.put(result)
    except Exception as e:
        queue.put(e)


def chroma_export(
    uri: str,
    collection: Optional[str] = None,
    limit: Optional[int] = -1,
    offset: Optional[int] = 0,
    batch_size: Optional[int] = 100,
    embed_feature: Optional[str] = "embedding",
    meta_features: Optional[List[str]] = None,
    id_feature: Optional[str] = "id",
    doc_feature: Optional[str] = "text_chunk",
    where: Optional[str] = None,
    where_document: Optional[str] = None,
    format_output: Optional[str] = "record",
    max_threads: Optional[int] = 1,
) -> Generator[Dict[str, Any], None, None]:
    """Exports data from ChromaDB."""
    parsed_uri = CDPUri.from_uri(uri)
    client = get_client_for_uri(parsed_uri)
    _collection = parsed_uri.collection or collection
    _batch_size = parsed_uri.batch_size or batch_size
    _offset = parsed_uri.offset or offset
    _limit = parsed_uri.limit or limit
    _start = _offset if _offset > 0 else 0
    chroma_collection = client.get_collection(_collection)
    col_count = chroma_collection.count()
    # precondition the DB for fetching data
    chroma_collection.get(limit=1, include=["embeddings"])  # noqa
    total_results_to_fetch = min(col_count, _limit) if _limit > 0 else col_count
    _where = None
    if where:
        _where = validate_where(json.loads(where))
    _where_document = None
    if where_document:
        _where_document = validate_where_document(json.loads(where_document))

    queue = Queue()
    num_batches = (total_results_to_fetch - _start + _batch_size - 1) // _batch_size
    fetched_results = 0
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for batch in range(num_batches):
            offset = _start + batch * _batch_size
            batch_limit = min(total_results_to_fetch - offset, _batch_size)
            executor.submit(
                read_large_data_in_chunks,
                collection=chroma_collection,
                queue=queue,
                offset=offset,
                limit=batch_limit,
                where=_where,
                where_document=_where_document,
            )

        while fetched_results < total_results_to_fetch:
            _results = queue.get()
            if _results is None:
                break
            if isinstance(_results, Exception):
                raise _results
            _results = _get_result_to_chroma_doc_list(_results)
            fetched_results += len(_results)
            if format_output == "record":
                _final_results = [r.model_dump() for r in _results]
            elif format_output == "jsonl":
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
            else:
                raise ValueError(f"Unsupported format: {format_output}")
            for _doc in _final_results:
                yield _doc


def chroma_export_cli(
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
    meta_features: Optional[List[str]] = typer.Option(
        None, help="The metadata features."
    ),
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
    format_output: Optional[str] = typer.Option(
        "record",
        "--format",
        help="Export format. Default is `record`. Supported formats: `record` and `jsonl`. ",
    ),
    max_threads: Optional[int] = typer.Option(
        1, "--max-threads", "-t", help="The maximum number of threads."
    ),
) -> None:
    if export_file and not append:
        with open(export_file, "w") as f:
            f.write("")
    if export_file:
        with open(export_file, "a") as f:
            for _doc in chroma_export(
                uri=uri,
                collection=collection,
                limit=limit,
                offset=offset,
                batch_size=batch_size,
                embed_feature=embed_feature,
                meta_features=meta_features,
                id_feature=id_feature,
                doc_feature=doc_feature,
                where=where,
                where_document=where_document,
                format_output=format_output,
                max_threads=max_threads,
            ):
                f.write(str(json.dumps(_doc)) + "\n")
    else:
        for _doc in chroma_export(
            uri=uri,
            collection=collection,
            limit=limit,
            offset=offset,
            batch_size=batch_size,
            embed_feature=embed_feature,
            meta_features=meta_features,
            id_feature=id_feature,
            doc_feature=doc_feature,
            where=where,
            where_document=where_document,
            format_output=format_output,
            max_threads=max_threads,
        ):
            typer.echo(json.dumps(_doc))
