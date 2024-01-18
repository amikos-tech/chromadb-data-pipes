import json
import sys
from typing import Annotated, Optional, List, Dict, Any

import typer

from chroma_dp import EmbeddableTextResource
from chroma_dp.utils.embedding import (
    SupportedEmbeddingFunctions,
    get_embedding_function_for_name,
)
from chroma_dp.utils.chroma import remap_features


def filter_embed(
    inf: typer.FileText = typer.Argument(sys.stdin),
    batch_size: Annotated[int, typer.Option(help="The batch size.")] = 100,
    embedding_function: Optional[SupportedEmbeddingFunctions] = typer.Option(
        ..., "--ef", help="The embedding function."
    ),
    embed_feature: Annotated[
        str, typer.Option(help="The embedding feature.")
    ] = "embedding",
    meta_features: Annotated[
        List[str], typer.Option(help="The metadata features.")
    ] = None,
    id_feature: Annotated[str, typer.Option(help="The id feature.")] = "id",
    doc_feature: Annotated[
        str, typer.Option(help="The document feature.")
    ] = "text_chunk",
):
    _batch: Dict[str, Any] = {
        "documents": [],
        "embeddings": [],
        "metadatas": [],
        "ids": [],
    }
    _embedding_function = get_embedding_function_for_name(embedding_function)
    for line in inf:
        doc = remap_features(
            json.loads(line),
            doc_feature,
            embed_feature=embed_feature,
            meta_features=meta_features,
            id_feature=id_feature,
        )
        _batch["documents"].append(doc.text_chunk)
        _batch["metadatas"].append(doc.metadata)
        _batch["ids"].append(doc.id)
        if len(_batch["documents"]) >= batch_size:
            _batch["embeddings"] = _embedding_function(_batch["documents"])
            for d, m, i, e in zip(
                _batch["documents"],
                _batch["metadatas"],
                _batch["ids"],
                _batch["embeddings"],
            ):
                typer.echo(
                    EmbeddableTextResource(
                        text_chunk=d, metadata=m, id=i, embedding=e
                    ).model_dump_json()
                )
            _batch: Dict[str, Any] = {
                "documents": [],
                "embeddings": [],
                "metadatas": [],
                "ids": [],
            }
    if len(_batch["documents"]) > 0:
        _batch["embeddings"] = _embedding_function(_batch["documents"])
        for d, m, i, e in zip(
            _batch["documents"],
            _batch["metadatas"],
            _batch["ids"],
            _batch["embeddings"],
        ):
            typer.echo(
                EmbeddableTextResource(
                    text_chunk=d, metadata=m, id=i, embedding=e
                ).model_dump_json()
            )
