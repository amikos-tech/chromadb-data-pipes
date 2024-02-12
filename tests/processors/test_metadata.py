import subprocess
import tempfile

import orjson as json

from chroma_dp import EmbeddableTextResource

cdp_cmd_args = ["python", "-m", "chroma_dp.main"]


def test_meta_process_add_metadata_stdin() -> None:
    embeddable_text_resource = EmbeddableTextResource(
        id="test_id", text_chunk="test_text", metadata=None, embedding=None
    )
    with tempfile.TemporaryFile() as input_file:
        input_file.write(json.dumps(embeddable_text_resource.model_dump()))
        input_file.write(b"\n")
        input_file.seek(0)
        result = subprocess.run(
            [*cdp_cmd_args, "meta", "-a", "key1=value1", "--attr", "key2=true"],
            stdin=input_file,
            capture_output=True,
        )
        assert result.returncode == 0
        doc = EmbeddableTextResource(**json.loads(result.stdout.decode()))
        assert doc.metadata is not None
        assert doc.metadata["key1"] == "value1"
        assert doc.metadata["key2"] is True


def test_meta_process_overwrite_metadata_stdin() -> None:
    embeddable_text_resource = EmbeddableTextResource(
        id="test_id",
        text_chunk="test_text",
        metadata={"key1": "old_value"},
        embedding=None,
    )
    with tempfile.TemporaryFile() as input_file:
        input_file.write(json.dumps(embeddable_text_resource.model_dump()))
        input_file.write(b"\n")
        input_file.seek(0)
        result = subprocess.run(
            [*cdp_cmd_args, "meta", "-a", "key1=new_value", "-o"],
            stdin=input_file,
            capture_output=True,
        )
        assert result.returncode == 0
        doc = EmbeddableTextResource(**json.loads(result.stdout.decode()))
        assert doc.metadata is not None
        assert doc.metadata["key1"] == "new_value"


def test_meta_process_overwrite_no_flag_metadata_stdin() -> None:
    """The metadata should not be overwritten if the --overwrite or -o flag is not set."""
    embeddable_text_resource = EmbeddableTextResource(
        id="test_id",
        text_chunk="test_text",
        metadata={"key1": "old_value"},
        embedding=None,
    )
    with tempfile.TemporaryFile() as input_file:
        input_file.write(json.dumps(embeddable_text_resource.model_dump()))
        input_file.write(b"\n")
        input_file.seek(0)
        result = subprocess.run(
            [*cdp_cmd_args, "meta", "-a", "key1=new_value"],
            stdin=input_file,
            capture_output=True,
        )
        assert result.returncode == 0
        doc = EmbeddableTextResource(**json.loads(result.stdout.decode()))
        assert doc.metadata is not None
        assert doc.metadata["key1"] == "old_value"


def test_meta_process_remove_metadata_stdin() -> None:
    embeddable_text_resource = EmbeddableTextResource(
        id="test_id",
        text_chunk="test_text",
        metadata={"key1": "value1", "key2": True},
        embedding=None,
    )
    with tempfile.TemporaryFile() as input_file:
        input_file.write(json.dumps(embeddable_text_resource.model_dump()))
        input_file.write(b"\n")
        input_file.seek(0)
        result = subprocess.run(
            [*cdp_cmd_args, "meta", "-k", "key1", "--remove-key", "key2"],
            stdin=input_file,
            capture_output=True,
        )
        assert result.returncode == 0
        doc = EmbeddableTextResource(**json.loads(result.stdout.decode()))
        assert doc.metadata is not None
        assert "key1" not in doc.metadata
        assert "key2" not in doc.metadata
