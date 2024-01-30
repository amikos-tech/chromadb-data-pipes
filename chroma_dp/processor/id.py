import hashlib
import json
import os
import sys
import uuid
from abc import ABC, abstractmethod
from typing import Any, Iterable, Annotated, Optional

import typer
from jinja2 import Environment
from overrides import EnforceOverrides, override
from ulid import ULID

from chroma_dp import EmbeddableTextResource, CdpProcessor
from chroma_dp.utils import smart_open


class IDStrategy(ABC, EnforceOverrides):
    @abstractmethod
    def generate_id(self, doc: EmbeddableTextResource) -> str:
        pass


class UUIDStrategy(IDStrategy):
    @override
    def generate_id(self, doc: EmbeddableTextResource) -> str:
        return str(uuid.uuid4())


class ULIDStrategy(IDStrategy):
    def __init__(self) -> None:
        self._ulid = ULID()

    @override
    def generate_id(self, doc: EmbeddableTextResource) -> str:
        return str(self._ulid.generate())


class ExprStrategy(IDStrategy):
    def __init__(self, expr: str) -> None:
        self._expr = expr
        self._ulid = ULID()
        self._jinja_env = Environment()

    @override
    def generate_id(self, doc: EmbeddableTextResource) -> str:
        return str(
            self._jinja_env.from_string(self._expr).render(
                uuid=uuid.uuid4, ulid=self._ulid.generate, **doc.model_dump()
            )
        )


class DocHashStrategy(IDStrategy):
    def __init__(self) -> None:
        self._sha256_hash = hashlib.sha256()

    @override
    def generate_id(self, doc: EmbeddableTextResource) -> str:
        if doc.text_chunk is None:
            raise ValueError("Document text chunk is None")
        self._sha256_hash.update(doc.text_chunk.encode("utf-8"))
        return self._sha256_hash.hexdigest()


class RandomHashStrategy(IDStrategy):
    def __init__(self) -> None:
        self._sha256_hash = hashlib.sha256()

    @override
    def generate_id(self, doc: EmbeddableTextResource) -> str:
        self._sha256_hash.update(os.urandom(32))
        return self._sha256_hash.hexdigest()


# --uuid --ulid --expr "{{ metadata.key }}" --doc-hash sha256
class IdStrategyGenerateProcessor(CdpProcessor[EmbeddableTextResource]):
    def __init__(
        self,
        strategy: IDStrategy = UUIDStrategy(),
    ):
        self._strategy = strategy

    def process(
        self, *, documents: Iterable[EmbeddableTextResource], **kwargs: Any
    ) -> Iterable[EmbeddableTextResource]:
        for doc in documents:
            doc.id = self._strategy.generate_id(doc)
            yield doc


def id_process(
    inf: typer.FileText = typer.Argument(sys.stdin),
    file: Optional[str] = typer.Option(None, "--in", help="The Chroma collection."),
    uuid: Annotated[
        Optional[bool],
        typer.Option(
            ...,
            "--uuid",
            "-u",
            help="Generate a UUID for each document.",
        ),
    ] = None,
    ulid: Annotated[
        Optional[bool],
        typer.Option(
            ...,
            "--ulid",
            "-U",
            help="Generate a ULID for each document.",
        ),
    ] = None,
    expr: Annotated[
        Optional[str],
        typer.Option(
            ...,
            "--expr",
            "-e",
            help="Generate id based on Jinja expression.",
        ),
    ] = None,
    doc_hash: Annotated[
        Optional[bool],
        typer.Option(
            ...,
            "--doc-hash",
            "-d",
            help="Generate ID based on resources text chunk hash.",
        ),
    ] = None,
    random_hash: Annotated[
        Optional[bool],
        typer.Option(
            ...,
            "--random-hash",
            "-r",
            help="Generate ID based on random hash.",
        ),
    ] = None,
) -> None:
    """Generates IDs for resources."""

    if not any([uuid, ulid, expr, doc_hash, random_hash]):
        typer.echo("Please specify an id generation strategy.")
        raise typer.Exit(code=1)

    if sum([bool(uuid), bool(ulid), bool(expr), bool(doc_hash), bool(random_hash)]) > 1:
        typer.echo("Please specify only one id generation strategy.")
        raise typer.Exit(code=1)
    strategy: Optional[IDStrategy] = None
    if uuid:
        strategy = UUIDStrategy()

    if ulid:
        strategy = ULIDStrategy()

    if expr:
        strategy = ExprStrategy(expr)

    if doc_hash:
        strategy = DocHashStrategy()

    if random_hash:
        strategy = RandomHashStrategy()

    if strategy is None:
        raise ValueError("Cannot find suitable strategy.")
    processor = IdStrategyGenerateProcessor(
        strategy=strategy,
    )

    def process_docs(in_line: str) -> None:
        doc = EmbeddableTextResource(**json.loads(in_line))
        for doc in processor.process(
            documents=[doc],
        ):
            typer.echo(json.dumps(doc.model_dump()))

    with smart_open(file, inf) as file_or_stdin:
        for line in file_or_stdin:
            process_docs(line)
