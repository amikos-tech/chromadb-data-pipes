import importlib
import sys
from typing import Optional, Dict, Any, Iterable, Annotated

import typer
from pydantic import BaseModel

from chroma_dp import CdpProducer, EmbeddableTextResource
from chroma_dp.utils import get_max_with_none


class QdrantUri(BaseModel):
    host: str
    port: int = 6333
    collection: Optional[str] = None
    doc_payload_field: Optional[str] = None
    limit: Optional[int] = -1
    offset: Optional[int] = 0
    batch_size: Optional[int] = 100

    @staticmethod
    def from_uri(uri: str) -> "QdrantUri":
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(uri)
        # if "qdrant" not in parsed.scheme:
        #     raise ValueError("The URI scheme must be `qdrant+http://`.")
        if not parsed.hostname:
            raise ValueError("The URI must have a hostname.")
        if not parsed.port:
            raise ValueError("The URI must have a port.")
        query = parse_qs(parsed.query)
        return QdrantUri(
            host=parsed.hostname,
            port=parsed.port or 6333,
            collection=parsed.path.lstrip("/") if parsed.path else None,
            doc_payload_field=query.get("doc_payload_field", [None])[0],  # type: ignore
            limit=int(query.get("limit", ["-1"])[0]),
            offset=int(query.get("offset", ["0"])[0]),
            batch_size=int(query.get("batch_size", ["100"])[0]),
        )


def convert_qdrant_record_to_chroma_resource(
    record: Any, doc_payload_field: str
) -> EmbeddableTextResource:
    return EmbeddableTextResource(
        id=str(record.id),
        text_chunk=record.payload[doc_payload_field],
        metadata={k: v for k, v in record.payload.items() if k != doc_payload_field},
        embedding=record.vector,
    )


class QdrantProducer(CdpProducer[EmbeddableTextResource]):
    def __init__(
        self,
        url: str,
        collection: Optional[str] = None,
        doc_payload_field: Optional[str] = None,
        batch_size: Optional[int] = 100,
    ) -> None:
        self.url = QdrantUri.from_uri(url)

        self.collection_name = collection or self.url.collection
        if not self.collection_name:
            raise ValueError("The collection name is required.")
        self.doc_payload_field = doc_payload_field or self.url.doc_payload_field
        if not self.doc_payload_field:  # do we need to enforce this?
            raise ValueError("The doc_payload_field is required.")
        self.batch_size = get_max_with_none(batch_size, self.url.batch_size)
        self.limit = self.url.limit
        self.offset = self.url.offset
        try:
            _qdrant = importlib.import_module("qdrant_client")
            self.qdrant = _qdrant.QdrantClient(host=self.url.host, port=self.url.port)
        except ImportError:
            raise ImportError(
                "The Qdrant Client is required. Please install it with `pip install qdrant-client`"
            )

        self.collection = self.qdrant.get_collection(
            collection_name=self.collection_name
        )

    def produce(
        self, limit: int = -1, offset: int = 0, **kwargs: Dict[str, Any]
    ) -> Iterable[EmbeddableTextResource]:
        _start = get_max_with_none(self.offset, offset)
        _limit = get_max_with_none(self.limit, limit)
        _buffer = []
        for _offset in range(_start, self.collection.vectors_count, self.batch_size):
            res = self.qdrant.scroll(
                collection_name=self.collection_name,
                offset=_offset,
                limit=_limit,
                with_vectors=True,
                with_payload=True,
            )
            _buffer.extend(res[0])
            if len(_buffer) >= self.batch_size:
                for doc in res[0][_offset : _offset + self.batch_size]:
                    yield convert_qdrant_record_to_chroma_resource(
                        doc, self.doc_payload_field  # type: ignore
                    )


def qdrant_export(
    uri: Annotated[
        str,
        typer.Argument(
            ...,
            help="Qdrant URI.",
        ),
    ],
    collection: Annotated[
        Optional[str],
        typer.Option(
            ...,
            "--collection",
            "-c",
            help="The collection name. The name can also be specified in the URI.",
        ),
    ] = None,
    doc_payload_field: Annotated[
        Optional[str],
        typer.Option(
            ...,
            "--doc",
            "-d",
            help="The payload field to use as the document. The field can also be specified in the URI.",
        ),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option(
            ...,
            "--batch-size",
            "-b",
            help="The batch size to use.",
        ),
    ] = 100,
    limit: Annotated[int, typer.Option(help="The limit.")] = -1,
    offset: Annotated[int, typer.Option(help="The offset.")] = 0,
) -> None:
    """Export PDF files from a directory to ChromaDB."""

    producer = QdrantProducer(
        url=uri,
        collection=collection,
        doc_payload_field=doc_payload_field,
        batch_size=batch_size,
    )
    _max = limit if limit > 0 else sys.maxsize
    count = 0
    start = offset
    for doc in producer.produce(limit=_max, offset=offset):
        if count < start:
            continue
        typer.echo(doc.model_dump_json())
        count += 1
        if count >= _max:
            break
