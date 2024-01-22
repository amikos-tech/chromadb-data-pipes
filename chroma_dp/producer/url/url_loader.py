import importlib
import sys
from typing import Optional, Callable, Dict, Any, Iterable, Annotated

import typer
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

from chroma_dp import CdpProducer, EmbeddableTextResource
from chroma_dp.processor.langchain_utils import convert_lc_doc_to_chroma_resource


class BSExtractor:
    def __init__(self, url: str, max_depth: int = 2) -> None:
        try:
            _bs = importlib.import_module("bs4")
            self.bs = _bs.BeautifulSoup
        except ImportError:
            raise ImportError(
                "The bs4 package is required. Please install it with `pip install bs4`"
            )
        self.url = url
        self.max_depth = max_depth

    def __call__(self, html: str) -> str:
        return str(self.bs(html, "html.parser").text)


class URLProducer(CdpProducer[EmbeddableTextResource]):
    def __init__(
        self,
        url: str,
        max_depth: int = 2,
        extractor: Optional[Callable[[str], str]] = None,
        batch_size: Optional[int] = 100,
    ) -> None:
        self.url = url
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.extractor = extractor or BSExtractor(url, max_depth)

    def produce(
        self, limit: int = -1, offset: int = 0, **kwargs: Dict[str, Any]
    ) -> Iterable[EmbeddableTextResource]:
        loader = RecursiveUrlLoader(
            url=self.url, max_depth=self.max_depth, extractor=self.extractor
        )
        docs = loader.load()
        _start = 0
        for _offset in range(_start, len(docs), self.batch_size):
            for doc in docs[_offset : _offset + self.batch_size]:
                yield convert_lc_doc_to_chroma_resource(doc)


def url_import(
    url: Annotated[
        str,
        typer.Argument(
            ...,
            help="The URL.",
        ),
    ],
    max_depth: Annotated[
        Optional[int],
        typer.Option(
            ...,
            "--max-depth",
            "-d",
            help="The maximum depth to crawl.",
        ),
    ] = 1,
    batch_size: Annotated[
        Optional[int],
        typer.Option(
            ...,
            "--batch-size",
            "-b",
            help="The batch size to use when processing the PDF files.",
        ),
    ] = 100,
    limit: Annotated[int, typer.Option(help="The limit.")] = -1,
    offset: Annotated[int, typer.Option(help="The offset.")] = 0,
) -> None:
    """Export PDF files from a directory to ChromaDB."""
    producer = URLProducer(url=url, max_depth=max_depth, batch_size=batch_size)
    start = offset
    count = 0
    max = limit if limit > 0 else sys.maxsize
    for doc in producer.produce():
        if count < start:
            continue
        typer.echo(doc.model_dump_json())
        count += 1
        if count >= max:
            break
