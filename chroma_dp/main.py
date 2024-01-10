import typer

from chroma_dp.huggingface import hf_commands

app = typer.Typer(no_args_is_help=True, help="ChromaDB Data Pump commands.")

app.add_typer(hf_commands, name="hf", no_args_is_help=True, help="Hugging Face commands.")

if __name__ == "__main__":
    app()
