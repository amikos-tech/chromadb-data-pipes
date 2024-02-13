import orjson as json
import sys
from typing import Any, Iterable, Annotated, Optional, List, Union, Dict

import typer
from jinja2 import Template

from chroma_dp import EmbeddableTextResource, CdpProcessor, Metadata
from chroma_dp.utils import smart_open
from chroma_dp.utils.templating import get_jinja_env


_jinja_env = get_jinja_env()


def process_value(value: str) -> Union[bool, float, int, str, Template]:
    """Parse the value as follows: bool>float>int>string"""
    if value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    if value.startswith("{{") and value.endswith("}}"):
        return _jinja_env.from_string(value)
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    elif "." in value:
        try:
            return float(value)
        except ValueError:
            pass
    else:
        try:
            return int(value)
        except ValueError:
            pass
    return value


TemplateMetadata = Dict[str, Union[str, int, float, bool, Template]]


class MetadataProcessor(CdpProcessor[EmbeddableTextResource]):
    def __init__(
        self,
        metadata: Optional[TemplateMetadata] = None,
        remove_keys: Optional[List[str]] = None,
        overwrite: bool = False,
    ):
        self._metadata = metadata
        self._remove_keys = remove_keys
        self._overwrite = overwrite

    def process(
        self, *, documents: Iterable[EmbeddableTextResource], **kwargs: Any
    ) -> Iterable[EmbeddableTextResource]:
        for doc in documents:
            if self._remove_keys and doc.metadata:
                for k in self._remove_keys:
                    if k in doc.metadata.keys():
                        del doc.metadata[k]
            if self._metadata:
                if not doc.metadata:
                    doc.metadata = {}
                for k, v in self._metadata.items():
                    if isinstance(v, Template):
                        # process rendered value
                        doc.metadata[k] = process_value(v.render(**doc.model_dump()))
                    else:
                        if self._overwrite or k not in doc.metadata.keys():
                            doc.metadata[k] = v
            yield doc


def meta_process(
    inf: typer.FileText = typer.Argument(sys.stdin),
    file: Optional[str] = typer.Option(None, "--in", help="The Chroma collection."),
    meta: Annotated[
        Optional[List[str]],
        typer.Option(
            ...,
            "--attr",
            "-a",
            "--meta",
            "-m",
            help="The metadata key value pairs to add e.g. key=value. "
            "The value will be parsed as follows: bool>float>int>string",
        ),
    ] = None,
    remove_keys: Annotated[
        Optional[List[str]],
        typer.Option(
            ...,
            "--remove-key",
            "-k",
            help="The keys to remove from the metadata.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            ...,
            "--overwrite",
            "-o",
            help="Indicates whether to overwrite the metadata if it already exists. Only applicable for --add.",
        ),
    ] = False,
) -> None:
    """Add or remove metadata."""
    kv_pairs: Metadata = {}
    if not meta and not remove_keys:
        typer.echo(
            "Please specify either --meta or --remove-key",
            err=True,
            color=typer.colors.RED,
            file=sys.stderr,
        )
        raise typer.Abort()
    if meta:
        for opt in meta:
            try:
                key, value = opt.split("=")
                kv_pairs[key] = process_value(value)
            except ValueError:
                typer.echo(
                    f"Invalid metadata: {opt}",
                    err=True,
                    color=typer.colors.RED,
                    file=sys.stderr,
                )
                raise typer.Abort()
    processor = MetadataProcessor(
        metadata=kv_pairs, remove_keys=remove_keys, overwrite=overwrite
    )

    def process_docs(in_line: str) -> None:
        doc = EmbeddableTextResource(**json.loads(in_line))
        for _doc in processor.process(
            documents=[doc],
        ):
            typer.echo(json.dumps(_doc.model_dump()))

    with smart_open(file, inf) as file_or_stdin:
        for line in file_or_stdin:
            process_docs(line)
