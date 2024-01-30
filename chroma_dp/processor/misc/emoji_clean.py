import json
import re
import sys
from typing import Optional, Iterable, Any

import typer

from chroma_dp import EmbeddableTextResource, CdpProcessor


def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


class EmojiCleanProcessor(CdpProcessor[EmbeddableTextResource]):
    def __init__(self, metadata_clean: Optional[bool] = False):
        self.metadata_clean = metadata_clean

    def process(
        self, *, documents: Iterable[EmbeddableTextResource], **kwargs: Any
    ) -> Iterable[EmbeddableTextResource]:
        for doc in documents:
            doc.text_chunk = remove_emojis(doc.text_chunk)
            if self.metadata_clean:
                for k, v in doc.metadata.items():
                    if isinstance(v, str):
                        doc.metadata[k] = remove_emojis(v)
            yield doc


def emoji_clean(
    inf: typer.FileText = typer.Argument(sys.stdin),
    file: Optional[str] = typer.Option(None, "--in", help="The Chroma collection."),
    metadata_clean: Optional[bool] = typer.Option(
        False, "--metadata-clean", "-m", help="Whether to clean the metadata too."
    ),
) -> None:
    """Chunk a document."""
    processor = EmojiCleanProcessor(metadata_clean=metadata_clean)

    def process_docs(line: str) -> None:
        doc = EmbeddableTextResource(**json.loads(line))
        for doc in processor.process(
            documents=[doc],
        ):
            typer.echo(json.dumps(doc.model_dump()))

    if file:
        with open(file, "r") as inf:
            for line in inf:
                process_docs(line)
    else:
        for line in inf:
            process_docs(line)
