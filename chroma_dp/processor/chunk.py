#
# class CdpProcessor(Protocol[D]):
#     def filter(
#             self, *, documents: Iterable[D], **kwargs: Dict[str, Any]
#     ) -> Iterable[D]:
#         ...
import json
import sys
from typing import Dict, Any, Iterable, Annotated, Optional

import typer

from chroma_dp import EmbeddableTextResource, CdpProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter

from chroma_dp.processor.langchain_utils import (
    convert_chroma_emb_resource_to_lc_doc,
    convert_lc_doc_to_chroma_resource,
)


class ChunkProcessor(CdpProcessor[EmbeddableTextResource]):
    def __init__(self, type: str = "character"):
        self.type = type

    def process(
        self, *, documents: Iterable[EmbeddableTextResource], **kwargs: Dict[str, Any]
    ) -> Iterable[EmbeddableTextResource]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=kwargs.get("size"),
            chunk_overlap=kwargs.get("overlap", 0),
            add_start_index=kwargs.get("add_start_index", False),
        )
        for doc in documents:
            split_docs = text_splitter.split_documents(
                [convert_chroma_emb_resource_to_lc_doc(doc)]
            )
            for split_doc in split_docs:
                yield convert_lc_doc_to_chroma_resource(split_doc, doc.metadata)


def chunk_process(
    size: Annotated[
        int,
        typer.Option(
            ...,
            "--size",
            "-s",
            help="The maximum size of each chunk",
        ),
    ],
    inf: typer.FileText = typer.Argument(sys.stdin),
    overlap: Annotated[
        int,
        typer.Option(
            ...,
            "--overlap",
            "-o",
            help="The overlap between chunks",
        ),
    ] = 0,
    add_start_index: Annotated[
        bool,
        typer.Option(
            ...,
            "--add-start-index",
            "-a",
            help="A flag whether to add the start index to the metadata",
        ),
    ] = False,
    type: Annotated[
        Optional[str],
        typer.Option(
            ...,
            "--type",
            "-t",
            help="The type of the chunking.",
        ),
    ] = "character",
) -> None:
    """Chunk a document."""
    processor = ChunkProcessor(type=type)
    for line in inf:
        doc = EmbeddableTextResource(**json.loads(line))
        for doc in processor.process(
            documents=[doc],
            size=size,
            overlap=overlap,
            add_start_index=add_start_index,
            type=type,
        ):
            typer.echo(doc.model_dump_json())
