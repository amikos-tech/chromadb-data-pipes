import typer
from chroma_dp.chroma.chroma_export import chroma_export
from chroma_dp.chroma.chroma_import import chroma_import
from chroma_dp.processor.chunk import chunk_process
from chroma_dp.processor.embed import filter_embed
from chroma_dp.huggingface import hf_import, hf_export
from chroma_dp.processor.id import id_process
from chroma_dp.processor.metadata import meta_process
from chroma_dp.processor.misc.emoji_clean import emoji_clean
from chroma_dp.producer.file.csv import csv_import
from chroma_dp.producer.file.pdf import pdf_import
from chroma_dp.producer.file.text import txt_import
from chroma_dp.producer.url.url_loader import url_import

app = typer.Typer(no_args_is_help=True, help="ChromaDB Data Pipes commands.")

# Import commands
import_commands = typer.Typer(no_args_is_help=True, help="Import commands.")

import_commands.command(
    name="pdf", help="Import PDF files from target dir.", no_args_is_help=True
)(pdf_import)

import_commands.command(
    name="url", help="Imports from remote url.", no_args_is_help=True
)(url_import)

import_commands.command(
    name="txt", help="Import text files from target dir.", no_args_is_help=True
)(txt_import)

import_commands.command(name="csv", help="Import csv file.", no_args_is_help=True)(
    csv_import
)

# Filter commands
transform_commands = typer.Typer(no_args_is_help=True, help="Transformer commands.")
transform_commands.command(
    name="emoji-clean",
    help="Cleans emojis from documents.",
    no_args_is_help=True,
)(emoji_clean)

app.add_typer(
    import_commands, name="imp", no_args_is_help=True, help="Import Commands."
)

app.add_typer(
    transform_commands, name="tx", no_args_is_help=True, help="Filter commands."
)

# Chunk commands

app.command(
    name="chunk",
    help="Chunk embeddable resources into smaller pieces.",
    no_args_is_help=True,
)(chunk_process)

# Embed commands

app.command(
    name="embed",
    help="Generate embeddings for embeddable resources.",
    no_args_is_help=True,
)(filter_embed)

# Chroma commands
app.command(
    name="export",
    help="Export data from ChromaDB.",
    no_args_is_help=True,
)(chroma_export)

app.command(
    name="import",
    help="Import data into ChromaDB.",
    no_args_is_help=True,
)(chroma_import)

## Dataset commands


app.command(
    name="ds-get",
    help="Gets a dataset from HF.",
    no_args_is_help=True,
)(hf_import)

app.command(
    name="ds-put",
    help="Upload a dataset to HF.",
    no_args_is_help=True,
)(hf_export)

## Metadata processor

app.command(
    name="meta",
    help="Add or remove metadata.",
    no_args_is_help=True,
)(meta_process)

## ID processor


app.command(
    name="id",
    help="Generate IDs for resources given a strategy.",
    no_args_is_help=True,
)(id_process)

if __name__ == "__main__":
    app()
