import json
import sys
from typing import Any, Iterable, Annotated, Optional

import typer

from chroma_dp import EmbeddableTextResource, CdpProcessor
from langchain.text_splitter import CharacterTextSplitter

from chroma_dp.processor.langchain_utils import (
    convert_chroma_emb_resource_to_lc_doc,
    convert_lc_doc_to_chroma_resource,
)


class ChunkProcessor(CdpProcessor[EmbeddableTextResource]):
    def __init__(self, type: Optional[str] = "character"):
        self.type = type

    def process(
        self, *, documents: Iterable[EmbeddableTextResource], **kwargs: Any
    ) -> Iterable[EmbeddableTextResource]:
        text_splitter = CharacterTextSplitter(
            separator=kwargs.get("separator") if kwargs.get("separator") else "\n",
            chunk_size=kwargs.get("size"),
            chunk_overlap=kwargs.get("overlap", 0),
            add_start_index=kwargs.get("add_start_index", False),
        )
        for doc in documents:
            split_docs = text_splitter.split_documents(
                [convert_chroma_emb_resource_to_lc_doc(doc)]
            )
            for _, split_doc in enumerate(split_docs):
                yield convert_lc_doc_to_chroma_resource(split_doc, doc.metadata)


def chunk_process(
    size: Annotated[
        int,
        typer.Option(
            ...,
            "--size",
            "-s",
            help="The maximum size of each chunk.",
        ),
    ],
    inf: typer.FileText = typer.Argument(sys.stdin),
    file: Optional[str] = typer.Option(None, "--in", help="The Chroma collection."),
    overlap: Annotated[
        int,
        typer.Option(
            ...,
            "--overlap",
            "-o",
            help="The overlap between chunks",
        ),
    ] = 0,
    separator: Annotated[
        str,
        typer.Option(
            ...,
            "--separator",
            "-p",
            help="The separator character to use for splitting the text.",
        ),
    ] = "\n",
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

    def process_docs(line: str) -> None:
        doc = EmbeddableTextResource(**json.loads(line))
        for doc in processor.process(
            documents=[doc],
            size=size,
            overlap=overlap,
            add_start_index=add_start_index,
            type=type,
            separator=separator,
        ):
            typer.echo(json.dumps(doc.model_dump()))

    if file:
        with open(file, "r") as inf:
            for line in inf:
                process_docs(line)
    else:
        for line in inf:
            process_docs(line)
