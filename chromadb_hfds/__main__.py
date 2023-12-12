from enum import Enum
from typing import Annotated, Optional, List

import typer
from chromadb.utils.embedding_functions import (
    ONNXMiniLM_L6_V2,
)

from chromadb_hfds.chroma_import import ChromaDBImportBuilder, ImportRequest

app = typer.Typer()


class SupportedEmbeddingFunctions(str, Enum):
    default = "default"


@app.command(name="import")  # type: ignore
def _import(
    dataset_id: Annotated[
        Optional[str],
        typer.Option(
            help="The Hugging Face dataset. Expected format: <user>/<dataset_id>."
        ),
    ],
    document_feature: Annotated[str, typer.Option(help="The document feature.")],
    chroma_endpoint: Annotated[str, typer.Option(help="The Chroma endpoint.")],
    collection: Annotated[str, typer.Option(help="The Chroma collection.")],
    dataset_split: Annotated[
        Optional[str], typer.Option(help="The Hugging Face")
    ] = "train",
    stream: Annotated[
        bool, typer.Option(help="Stream dataset instead of downloading.")
    ] = False,
    create: Annotated[
        bool, typer.Option(help="Create the Chroma collection if it does not exist.")
    ] = False,
    embedding_feature: Annotated[
        str, typer.Option(help="The embedding feature.")
    ] = None,
    metadata_features: Annotated[
        List[str], typer.Option(help="The metadata features.")
    ] = None,
    id_feature: Annotated[str, typer.Option(help="The id feature.")] = None,
    limit: Annotated[int, typer.Option(help="The limit.")] = -1,
    offset: Annotated[int, typer.Option(help="The offset.")] = 0,
    batch_size: Annotated[int, typer.Option(help="The batch size.")] = 100,
    embedding_function: Annotated[
        SupportedEmbeddingFunctions, typer.Option(help="The embedding function.")
    ] = SupportedEmbeddingFunctions.default,
) -> None:
    """Import a Hugging Face dataset into ChromaDB."""

    if embedding_function == SupportedEmbeddingFunctions.default:
        _embedding_function = ONNXMiniLM_L6_V2()
    else:
        raise ValueError(f"Unsupported embedding function: {embedding_function}")
    import_request = ImportRequest(
        dataset=dataset_id,
        dataset_split=dataset_split,
        dataset_stream=stream,
        endpoint=chroma_endpoint,
        collection=collection,
        create=create,
        document_feature=document_feature,
        embedding_feature=embedding_feature,
        metadata_features=metadata_features,
        id_feature=id_feature,
        limit=limit,
        offset=offset,
        batch_size=batch_size,
        embedding_function=_embedding_function,
        client=None,
    )
    ChromaDBImportBuilder(import_request).run()


if __name__ == "__main__":
    app()
