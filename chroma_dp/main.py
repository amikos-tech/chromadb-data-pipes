import typer
from chroma_dp.chroma.chroma_export import chroma_export
from chroma_dp.chroma.chroma_import import chroma_import
from chroma_dp.transformers.embed import filter_embed
from chroma_dp.huggingface import hf_import, hf_export

app = typer.Typer(no_args_is_help=True, help="ChromaDB Data Pump commands.")

# Import commands
import_commands = typer.Typer(no_args_is_help=True, help="Import commands.")
import_commands.command(name="chroma", help="Chroma Import.", no_args_is_help=True)(chroma_import)
import_commands.command(name="hf", help="Import HuggingFace Dataset", no_args_is_help=True)(hf_import)

# Export commands
export_commands = typer.Typer(no_args_is_help=True, help="Export commands.")
export_commands.command(name="chroma", help="Imports data into ChromaDB.", no_args_is_help=True)(chroma_export)
export_commands.command(name="hf", help="Export data to HuggingFace Dataset", no_args_is_help=True)(hf_export)
# Filter commands
filter_commands = typer.Typer(no_args_is_help=True)
transform_commands = typer.Typer(no_args_is_help=True, help="Transformer commands.")
transform_commands.command(name="embed", help="Embedding Transformer. Re-embeds document contents.",
                           no_args_is_help=True)(filter_embed)

app.add_typer(export_commands, name="exp", no_args_is_help=True, help="Export Commands.")
app.add_typer(import_commands, name="imp", no_args_is_help=True, help="Import Commands.")
app.add_typer(filter_commands, name="flt", no_args_is_help=True, help="Filter commands.")
app.add_typer(transform_commands, name="tx", no_args_is_help=True, help="Filter commands.")

if __name__ == "__main__":
    app()
