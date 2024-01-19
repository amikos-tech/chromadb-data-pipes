from typing import Dict, Any, Iterable, Optional, Annotated

import typer
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from chroma_dp import CdpProducer, EmbeddableTextResource
from chroma_dp.processor.langchain_utils import convert_lc_doc_to_chroma_resource


class LangchainTXTProducer(CdpProducer[EmbeddableTextResource]):
    def __init__(
        self,
        path: str,
        glob: Optional[str] = None,
        recursive: bool = False,
        batch_size: int = 100,
    ) -> None:
        self.path = path
        self.glob = glob
        self.recursive = recursive
        self.batch_size = batch_size

    def produce(
        self, limit: int = -1, offset: int = 0, **kwargs: Dict[str, Any]
    ) -> Iterable[EmbeddableTextResource]:
        loader = DirectoryLoader(
            path=self.path,
            glob=self.glob,
            loader_cls=TextLoader,
            recursive=self.recursive,
        )
        docs = loader.load()
        _start = 0
        for _offset in range(_start, len(docs), self.batch_size):
            for doc in docs[_offset : _offset + self.batch_size]:
                yield convert_lc_doc_to_chroma_resource(doc)


def txt_import(
    path: Annotated[
        str,
        typer.Argument(
            ...,
            help="The path to the directory containing the text files.",
        ),
    ],
    glob: Annotated[
        Optional[str],
        typer.Option(
            ...,
            "--glob",
            "-g",
            help="The glob pattern to filter files in the directory.",
        ),
    ] = "**/[!.]*.md",
    recursive: Annotated[
        Optional[bool],
        typer.Option(
            ...,
            "--recursive",
            "-r",
            help="A flag whether to recursively search the directory.",
        ),
    ] = False,
    batch_size: Annotated[
        Optional[int],
        typer.Option(
            ...,
            "--batch-size",
            "-b",
            help="The batch size to use when processing the text files.",
        ),
    ] = 100,
) -> None:
    """Export text files from a directory to ChromaDB."""
    producer = LangchainTXTProducer(
        path=path, glob=glob, recursive=recursive, batch_size=batch_size
    )
    for doc in producer.produce():
        typer.echo(doc.model_dump_json())
