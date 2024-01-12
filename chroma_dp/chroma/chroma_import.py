import json
import sys
import uuid
from typing import Annotated, Optional, List, Dict, Any

import typer
from chromadb import EmbeddingFunction
from chromadb.api.models import Collection
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from chroma_dp import ChromaDocument
from chroma_dp.huggingface import SupportedEmbeddingFunctions
from chroma_dp.utils.chroma import CDPUri, get_client_for_uri, remap_features


def add_to_col(col: Collection, batch: Dict[str, Any], upsert: bool = False, ef: EmbeddingFunction = None) -> None:
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
        uri: Annotated[str, typer.Option(help="The Chroma endpoint.")],
        collection: Annotated[str, typer.Option(help="The Chroma collection.")] = None,
        inf: typer.FileText = typer.Argument(sys.stdin),
        import_file: Optional[str] = typer.Option(None, "--in", help="The Chroma collection."),
        create: Annotated[
            bool, typer.Option(help="Create the Chroma collection if it does not exist.")
        ] = False,
        limit: Annotated[int, typer.Option(help="The limit.")] = -1,
        offset: Annotated[int, typer.Option(help="The offset.")] = 0,
        batch_size: Annotated[int, typer.Option(help="The batch size.")] = 100,
        embedding_function: Optional[SupportedEmbeddingFunctions] = typer.Option(None,
                                                                                 "--ef",
                                                                                 help="The embedding function."),
        upsert: Annotated[bool, typer.Option(help="Upsert documents.")] = False,
        embed_feature: Annotated[
            str, typer.Option(help="The embedding feature.")
        ] = "embedding",
        meta_features: Annotated[
            List[str], typer.Option(help="The metadata features.")
        ] = None,
        id_feature: Annotated[str, typer.Option(help="The id feature.")] = "id",
        doc_feature: Annotated[str, typer.Option(help="The document feature.")] = "text_chunk",
):
    _embedding_function = None
    if embedding_function == SupportedEmbeddingFunctions.default:
        _embedding_function = ONNXMiniLM_L6_V2()
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
    chroma_collection = client.get_or_create_collection(_collection)
    lc_count = 0
    if import_file:
        with open(import_file, 'r') as inf:
            for line in inf:
                if lc_count < _offset:
                    continue
                if _limit != -1 and lc_count >= _limit:
                    break
                doc = remap_features(json.loads(line), doc_feature, embed_feature, meta_features, id_feature)
                _batch["documents"].append(doc.text_chunk)
                _batch["embeddings"].append(
                    doc.embedding if _embedding_function is None else None)  # call EF?
                _batch["metadatas"].append(doc.metadata)
                _batch["ids"].append(doc.id if doc.id else uuid.uuid4())
                if len(_batch["documents"]) >= _batch_size:
                    add_to_col(chroma_collection, _batch, _upsert, _embedding_function)
                    _batch = {
                        "documents": [],
                        "embeddings": [],
                        "metadatas": [],
                        "ids": [],
                    }
                lc_count += 1
        if len(_batch["documents"]) > 0:
            add_to_col(chroma_collection, _batch, _upsert, _embedding_function)
    else:
        for line in inf:
            if lc_count < _offset:
                continue
            if _limit != -1 and lc_count >= _limit:
                break
            doc = remap_features(json.loads(line), doc_feature, embed_feature, meta_features, id_feature)
            _batch["documents"].append(doc.text_chunk)
            _batch["embeddings"].append(
                doc.embedding if _embedding_function is None else None)  # call EF?
            _batch["metadatas"].append(doc.metadata)
            _batch["ids"].append(doc.id if doc.id else uuid.uuid4())
            if len(_batch["documents"]) >= _batch_size:
                add_to_col(chroma_collection, _batch, _upsert, _embedding_function)
                _batch = {
                    "documents": [],
                    "embeddings": [],
                    "metadatas": [],
                    "ids": [],
                }
            lc_count += 1
    if len(_batch["documents"]) > 0:
        add_to_col(chroma_collection, _batch, _upsert, _embedding_function)
