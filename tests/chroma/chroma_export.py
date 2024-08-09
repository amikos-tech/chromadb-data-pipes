import os
import subprocess
from os.path import abspath

import orjson as json
from testcontainers.chroma import ChromaContainer

from chroma_dp import EmbeddableTextResource

cdp_cmd_args = ["python", "-m", "chroma_dp.main"]


def test_export_record() -> None:
    result = subprocess.run(
        [
            *cdp_cmd_args,
            "export",
            "file://./sample-data/chroma/chroma-data-single/test_collection",
            "--format",
            "record",
            "--limit",
            "1",
        ],
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
        [
            *cdp_cmd_args,
            "export",
            "file://./sample-data/chroma/chroma-data-single/test_collection",
            "--limit",
            "1",
        ],
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
        [
            *cdp_cmd_args,
            "export",
            "file://./sample-data/chroma/chroma-data-single/test_collection",
            "--format",
            "jsonl",
            "--limit",
            "1",
        ],
        capture_output=True,
    )
    assert result.returncode == 0
    doc = json.loads(result.stdout.decode())
    assert doc is not None
    assert doc["a"] is not None
    assert doc["text_chunk"] is not None
    assert doc["embedding"] is not None
    assert doc["id"] is not None


def test_export_remote() -> None:
    with ChromaContainer().with_volume_mapping(
        abspath("../../sample-data/chroma/chroma-data-single/"), "/chroma/chroma", "rw"
    ) as c:
        result = subprocess.run(
            [
                *cdp_cmd_args,
                "export",
                f"http://{c.get_config()['endpoint']}/test_collection",
                "--format",
                "jsonl",
                "--limit",
                "1",
            ],
            capture_output=True,
        )
        assert result.returncode == 0
        doc = json.loads(result.stdout.decode())
        assert doc is not None
        assert doc["a"] is not None
        assert doc["text_chunk"] is not None
        assert doc["embedding"] is not None
        assert doc["id"] is not None


def test_export_remote_with_env_token_auth() -> None:
    with (
        ChromaContainer()
        .with_env("CHROMA_SERVER_AUTHN_CREDENTIALS", "chr0ma-t0k3n")
        .with_env(
            "CHROMA_SERVER_AUTHN_PROVIDER",
            "chromadb.auth.token_authn.TokenAuthenticationServerProvider",
        )
        .with_volume_mapping(
            abspath("../../sample-data/chroma/chroma-data-single/"),
            "/chroma/chroma",
            "rw",
        ) as c
    ):
        os.environ["CHROMA_TOKEN_AUTH"] = "chr0ma-t0k3n"
        result = subprocess.run(
            [
                *cdp_cmd_args,
                "export",
                f"http://{c.get_config()['endpoint']}/test_collection",
                "--format",
                "jsonl",
                "--limit",
                "1",
            ],
            env=os.environ,
            capture_output=True,
        )
        assert result.returncode == 0
        doc = json.loads(result.stdout.decode())
        assert doc is not None
        assert doc["a"] is not None
        assert doc["text_chunk"] is not None
        assert doc["embedding"] is not None
        assert doc["id"] is not None


def test_export_remote_with_uri_token_auth() -> None:
    with (
        ChromaContainer()
        .with_env("CHROMA_SERVER_AUTHN_CREDENTIALS", "chr0ma-t0k3n")
        .with_env(
            "CHROMA_SERVER_AUTHN_PROVIDER",
            "chromadb.auth.token_authn.TokenAuthenticationServerProvider",
        )
        .with_volume_mapping(
            abspath("../../sample-data/chroma/chroma-data-single/"),
            "/chroma/chroma",
            "rw",
        ) as c
    ):
        result = subprocess.run(
            [
                *cdp_cmd_args,
                "export",
                f"http://__auth_token__:chr0ma-t0k3n@{c.get_config()['endpoint']}/test_collection",
                "--format",
                "jsonl",
                "--limit",
                "1",
            ],
            capture_output=True,
        )
        assert result.returncode == 0
        doc = json.loads(result.stdout.decode())
        assert doc is not None
        assert doc["a"] is not None
        assert doc["text_chunk"] is not None
        assert doc["embedding"] is not None
        assert doc["id"] is not None


def test_export_remote_with_env_xtoken_auth() -> None:
    with (
        ChromaContainer()
        .with_env("CHROMA_AUTH_TOKEN_TRANSPORT_HEADER", "X-Chroma-Token")
        .with_env("CHROMA_SERVER_AUTHN_CREDENTIALS", "chr0ma-t0k3n")
        .with_env(
            "CHROMA_SERVER_AUTHN_PROVIDER",
            "chromadb.auth.token_authn.TokenAuthenticationServerProvider",
        )
        .with_volume_mapping(
            abspath("../../sample-data/chroma/chroma-data-single/"),
            "/chroma/chroma",
            "rw",
        ) as c
    ):
        os.environ["CHROMA_XTOKEN_AUTH"] = "chr0ma-t0k3n"
        result = subprocess.run(
            [
                *cdp_cmd_args,
                "export",
                f"http://{c.get_config()['endpoint']}/test_collection",
                "--format",
                "jsonl",
                "--limit",
                "1",
            ],
            env=os.environ,
            capture_output=True,
        )
        assert result.returncode == 0
        doc = json.loads(result.stdout.decode())
        assert doc is not None
        assert doc["a"] is not None
        assert doc["text_chunk"] is not None
        assert doc["embedding"] is not None
        assert doc["id"] is not None


def test_export_remote_with_uri_xtoken_auth() -> None:
    with (
        ChromaContainer()
        .with_env("CHROMA_AUTH_TOKEN_TRANSPORT_HEADER", "X-Chroma-Token")
        .with_env("CHROMA_SERVER_AUTHN_CREDENTIALS", "chr0ma-t0k3n")
        .with_env(
            "CHROMA_SERVER_AUTHN_PROVIDER",
            "chromadb.auth.token_authn.TokenAuthenticationServerProvider",
        )
        .with_volume_mapping(
            abspath("../../sample-data/chroma/chroma-data-single/"),
            "/chroma/chroma",
            "rw",
        ) as c
    ):
        result = subprocess.run(
            [
                *cdp_cmd_args,
                "export",
                f"http://__x_chroma_token__:chr0ma-t0k3n@{c.get_config()['endpoint']}/test_collection",
                "--format",
                "jsonl",
                "--limit",
                "1",
            ],
            capture_output=True,
        )
        assert result.returncode == 0
        doc = json.loads(result.stdout.decode())
        assert doc is not None
        assert doc["a"] is not None
        assert doc["text_chunk"] is not None
        assert doc["embedding"] is not None
        assert doc["id"] is not None


def test_export_remote_with_env_basic_auth() -> None:
    with (
        ChromaContainer()
        .with_env("CHROMA_SERVER_AUTHN_CREDENTIALS_FILE", "server.htpasswd")
        .with_env(
            "CHROMA_SERVER_AUTHN_PROVIDER",
            "chromadb.auth.basic_authn.BasicAuthenticationServerProvider",
        )
        .with_volume_mapping(
            abspath("../../sample-data/chroma/chroma-data-single/"),
            "/chroma/chroma",
            "rw",
        )
        .with_volume_mapping(
            abspath("../../sample-data/server.htpasswd"),
            "/chroma/server.htpasswd",
            "rw",
        ) as c
    ):
        os.environ["CHROMA_BASIC_AUTH"] = "admin:admin"
        result = subprocess.run(
            [
                *cdp_cmd_args,
                "export",
                f"http://{c.get_config()['endpoint']}/test_collection",
                "--format",
                "jsonl",
                "--limit",
                "1",
            ],
            env=os.environ,
            capture_output=True,
        )
        assert result.returncode == 0
        doc = json.loads(result.stdout.decode())
        assert doc is not None
        assert doc["a"] is not None
        assert doc["text_chunk"] is not None
        assert doc["embedding"] is not None
        assert doc["id"] is not None


def test_export_remote_with_uri_basic_auth() -> None:
    with (
        ChromaContainer()
        .with_env("CHROMA_SERVER_AUTHN_CREDENTIALS_FILE", "server.htpasswd")
        .with_env(
            "CHROMA_SERVER_AUTHN_PROVIDER",
            "chromadb.auth.basic_authn.BasicAuthenticationServerProvider",
        )
        .with_volume_mapping(
            abspath("../../sample-data/chroma/chroma-data-single/"),
            "/chroma/chroma",
            "rw",
        )
        .with_volume_mapping(
            abspath("../../sample-data/server.htpasswd"),
            "/chroma/server.htpasswd",
            "rw",
        ) as c
    ):
        result = subprocess.run(
            [
                *cdp_cmd_args,
                "export",
                f"http://admin:admin@{c.get_config()['endpoint']}/test_collection",
                "--format",
                "jsonl",
                "--limit",
                "1",
            ],
            capture_output=True,
        )
        assert result.returncode == 0
        doc = json.loads(result.stdout.decode())
        assert doc is not None
        assert doc["a"] is not None
        assert doc["text_chunk"] is not None
        assert doc["embedding"] is not None
        assert doc["id"] is not None
