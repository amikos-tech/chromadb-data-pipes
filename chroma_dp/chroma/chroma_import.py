import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Optional, List, Dict, Any

import typer
from chromadb import EmbeddingFunction
from chromadb.api.models import Collection

from chroma_dp.utils import smart_open
from chroma_dp.utils.embedding import (
    SupportedEmbeddingFunctions,
    get_embedding_function_for_name,
)
from chroma_dp.utils.chroma import (
    CDPUri,
    get_client_for_uri,
    remap_features,
    DistanceFunction,
)


def add_to_col(
    col: Collection,
    batch: Dict[str, Any],
    upsert: bool = False,
    ef: EmbeddingFunction = None,
) -> None:
    try:
        if ef:
            batch["embeddings"] = ef(batch["documents"])
        if upsert:
            col.upsert(**batch)
        else:
            col.add(**batch)
    except Exception as e:
        raise e


def chroma_import(
    uri: Annotated[str, typer.Argument(help="The Chroma endpoint.")],
    collection: Annotated[
        Optional[str], typer.Option(help="The Chroma collection.")
    ] = None,
    inf: typer.FileText = typer.Argument(
        sys.stdin, help="Stdin input. Requires to pass `-` as the argument."
    ),
    import_file: Optional[str] = typer.Option(
        None, "--in", help="The file to use for the import instead of stdin."
    ),
    create: Annotated[
        bool, typer.Option(help="Create the Chroma collection if it does not exist.")
    ] = False,
    limit: Annotated[int, typer.Option(help="The limit.")] = -1,
    offset: Annotated[int, typer.Option(help="The offset.")] = 0,
    batch_size: Annotated[int, typer.Option(help="The batch size.")] = 100,
    embedding_function: Optional[SupportedEmbeddingFunctions] = typer.Option(
        None, "--ef", help="The embedding function."
    ),
    upsert: Annotated[bool, typer.Option(help="Upsert documents.")] = False,
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
    distance_function: Optional[DistanceFunction] = typer.Option(
        None,
        "--df",
        help="The distance function to use when creating a collection",
    ),
    max_threads: Optional[int] = typer.Option(
        1, "--max-threads", "-t", help="The maximum number of threads."
    ),
) -> None:
    _embedding_function = None
    if embedding_function is not None:
        _embedding_function = get_embedding_function_for_name(embedding_function)
    if uri is None:
        raise ValueError("Please provide a ChromaDP URI.")
    parsed_uri = CDPUri.from_uri(uri)
    client = get_client_for_uri(parsed_uri)
    _collection = parsed_uri.collection or collection
    _batch_size = parsed_uri.batch_size or batch_size
    _offset = parsed_uri.offset or offset
    _limit = parsed_uri.limit or limit
    _upsert = parsed_uri.upsert or upsert
    _create = parsed_uri.create_collection or create
    _batch: Dict[str, Any] = {
        "documents": [],
        "embeddings": [],
        "metadatas": [],
        "ids": [],
    }
    _distance_function = (
        distance_function or parsed_uri.distance_function or DistanceFunction.l2
    )
    if _create:
        chroma_collection = client.get_or_create_collection(
            _collection, metadata={"hnsw:space": _distance_function.value}
        )
    else:
        chroma_collection = client.get_collection(_collection)
    lc_count = 0
    with smart_open(import_file, inf) as file_or_stdin:
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            for line in file_or_stdin:
                if lc_count < _offset:
                    continue
                if _limit != -1 and lc_count >= _limit:
                    break
                doc = remap_features(
                    json.loads(line),
                    doc_feature,
                    embed_feature,
                    meta_features,
                    id_feature,
                )
                _batch["documents"].append(doc.text_chunk)
                _batch["embeddings"].append(
                    doc.embedding if _embedding_function is None else None
                )  # call EF?
                _batch["metadatas"].append(doc.metadata)
                _batch["ids"].append(doc.id if doc.id else uuid.uuid4())
                if len(_batch["documents"]) >= _batch_size:
                    executor.submit(
                        add_to_col,
                        chroma_collection,
                        _batch,
                        _upsert,
                        _embedding_function,
                    )
                    _batch = {
                        "documents": [],
                        "embeddings": [],
                        "metadatas": [],
                        "ids": [],
                    }
                lc_count += 1
            if len(_batch["documents"]) > 0:
                executor.submit(
                    add_to_col, chroma_collection, _batch, _upsert, _embedding_function
                )
