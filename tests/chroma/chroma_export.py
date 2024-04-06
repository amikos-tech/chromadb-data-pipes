import subprocess

import orjson as json

from chroma_dp import EmbeddableTextResource

cdp_cmd_args = ["python", "-m", "chroma_dp.main"]


def test_export_record() -> None:
    result = subprocess.run(
        [*cdp_cmd_args, "export", "file://./sample-data/chroma/chroma-data-single/test_collection", "--format",
         "record", "--limit", "1"],
        capture_output=True,
    )
    assert result.returncode == 0
    doc = EmbeddableTextResource(**json.loads(result.stdout.decode()))
    assert doc.metadata is not None
    assert doc.metadata["a"] is not None
    assert doc.text_chunk is not None
    assert doc.embedding is not None
    assert doc.id is not None


def test_export_default_record() -> None:
    result = subprocess.run(
        [*cdp_cmd_args, "export", "file://./sample-data/chroma/chroma-data-single/test_collection", "--limit", "1"],
        capture_output=True,
    )
    assert result.returncode == 0
    doc = EmbeddableTextResource(**json.loads(result.stdout.decode()))
    assert doc.metadata is not None
    assert doc.metadata["a"] is not None
    assert doc.text_chunk is not None
    assert doc.embedding is not None
    assert doc.id is not None


def test_export_jsonl() -> None:
    result = subprocess.run(
        [*cdp_cmd_args, "export", "file://./sample-data/chroma/chroma-data-single/test_collection", "--format", "jsonl",
         "--limit", "1"],
        capture_output=True,
    )
    assert result.returncode == 0
    doc = json.loads(result.stdout.decode())
    assert doc is not None
    assert doc["a"] is not None
    assert doc["text_chunk"] is not None
    assert doc["embedding"] is not None
    assert doc["id"] is not None
