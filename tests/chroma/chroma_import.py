import subprocess
import tempfile

import chromadb
import orjson as json

cdp_cmd_args = ["python", "-m", "chroma_dp.main"]


def test_import_with_metadata_features() -> None:
    with tempfile.TemporaryDirectory() as tdir:
        with tempfile.TemporaryFile() as input_file:
            input_file.write(
                json.dumps(
                    {
                        "id": "test",
                        "text_chunk": "test",
                        "a": "test",
                        "embedding": [1, 2, 3],
                    }
                )
            )
            input_file.write(b"\n")
            input_file.seek(0)
            result = subprocess.run(
                [
                    *cdp_cmd_args,
                    "import",
                    f"file://{tdir}/test_collection",
                    "--create",
                    "--meta-features",
                    "a",
                ],
                stdin=input_file,
                capture_output=True,
            )
            assert result.returncode == 0
            client = chromadb.PersistentClient(path=tdir)
            col = client.get_collection("test_collection")
            records = col.get(ids=["test"])
            assert len(records["metadatas"]) == 1
            assert records["metadatas"][0]["a"] == "test"


def test_import_with_metadata_features_short() -> None:
    with tempfile.TemporaryDirectory() as tdir:
        with tempfile.TemporaryFile() as input_file:
            input_file.write(
                json.dumps(
                    {
                        "id": "test",
                        "text_chunk": "test",
                        "a": "test",
                        "embedding": [1, 2, 3],
                    }
                )
            )
            input_file.write(b"\n")
            input_file.seek(0)
            result = subprocess.run(
                [
                    *cdp_cmd_args,
                    "import",
                    f"file://{tdir}/test_collection",
                    "--create",
                    "-m",
                    "a",
                ],
                stdin=input_file,
                capture_output=True,
            )
            assert result.returncode == 0
            client = chromadb.PersistentClient(path=tdir)
            col = client.get_collection("test_collection")
            records = col.get(ids=["test"])
            assert len(records["metadatas"]) == 1
            assert records["metadatas"][0]["a"] == "test"
