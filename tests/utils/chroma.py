import os

from chroma_dp.utils.chroma import CDPUri, _get_auth_settings_for_uri


def test_parse_cdp_uri_basic() -> None:
    uri = (
        "https://basic_user:basic_password@localhost:5432"
        "/mycollection?tenant=mytenant&database=mydb&batch_size=100&limit=10&offset=0"
    )
    parsed = CDPUri.from_uri(uri)
    assert parsed.host_or_path == "localhost"
    assert parsed.port == 5432
    assert parsed.database == "mydb"
    assert parsed.collection == "mycollection"
    assert parsed.tenant == "mytenant"
    assert parsed.batch_size == 100
    assert parsed.auth is not None
    assert parsed.auth["type"] == "basic"
    assert parsed.auth["username"] == "basic_user"
    assert parsed.auth["password"] == "basic_password"


def test_parse_cdp_uri_local_absolute() -> None:
    uri = "file:///abs/path/persist_dir/some_collection?tenant=mytenant&database=db1&batch_size=100&limit=10&offset=0"
    parsed = CDPUri.from_uri(uri)
    assert parsed.is_local is True
    assert parsed.host_or_path == "/abs/path/persist_dir/"
    assert parsed.tenant == "mytenant"
    assert parsed.database == "db1"
    assert parsed.collection == "some_collection"


def test_parse_cdp_uri_local_relative() -> None:
    uri = "file://abs/path/persist_dir/some_collection?tenant=mytenant&database=db1&batch_size=100&limit=10&offset=0"
    parsed = CDPUri.from_uri(uri)
    assert parsed.is_local is True
    assert parsed.host_or_path == "abs/path/persist_dir/"
    assert parsed.tenant == "mytenant"
    assert parsed.database == "db1"
    assert parsed.collection == "some_collection"


def test_parse_url_with_env_basic_auth() -> None:
    os.environ["CHROMA_BASIC_AUTH"] = "basic_user:basic_password"
    uri = (
        "https://localhost:5432"
        "/mycollection?tenant=mytenant&database=mydb&batch_size=100&limit=10&offset=0"
    )
    parsed = CDPUri.from_uri(uri)
    assert parsed.host_or_path == "localhost"
    assert parsed.port == 5432
    assert parsed.database == "mydb"
    assert parsed.collection == "mycollection"
    assert parsed.tenant == "mytenant"
    assert parsed.batch_size == 100
    settings = _get_auth_settings_for_uri(parsed)
    assert settings is not None
    assert (
        settings["chroma_client_auth_provider"]
        == "chromadb.auth.basic_authn.BasicAuthClientProvider"
    )
    assert settings["chroma_client_auth_credentials"] == os.getenv("CHROMA_BASIC_AUTH")


def test_parse_url_with_uri_basic_auth() -> None:
    creds = "basic_user:basic_password"
    uri = (
        f"https://{creds}@localhost:5432"
        "/mycollection?tenant=mytenant&database=mydb&batch_size=100&limit=10&offset=0"
    )
    parsed = CDPUri.from_uri(uri)
    assert parsed.host_or_path == "localhost"
    assert parsed.port == 5432
    assert parsed.database == "mydb"
    assert parsed.collection == "mycollection"
    assert parsed.tenant == "mytenant"
    assert parsed.batch_size == 100
    settings = _get_auth_settings_for_uri(parsed)
    assert settings is not None
    assert (
        settings["chroma_client_auth_provider"]
        == "chromadb.auth.basic_authn.BasicAuthClientProvider"
    )
    assert settings["chroma_client_auth_credentials"] == creds


def test_parse_url_with_env_token_auth() -> None:
    os.environ["CHROMA_TOKEN_AUTH"] = "token"
    uri = (
        "https://localhost:5432"
        "/mycollection?tenant=mytenant&database=mydb&batch_size=100&limit=10&offset=0"
    )
    parsed = CDPUri.from_uri(uri)
    assert parsed.host_or_path == "localhost"
    assert parsed.port == 5432
    assert parsed.database == "mydb"
    assert parsed.collection == "mycollection"
    assert parsed.tenant == "mytenant"
    assert parsed.batch_size == 100
    settings = _get_auth_settings_for_uri(parsed)
    assert settings is not None
    assert (
        settings["chroma_client_auth_provider"]
        == "chromadb.auth.token_authn.TokenAuthClientProvider"
    )
    assert settings["chroma_client_auth_credentials"] == os.getenv("CHROMA_TOKEN_AUTH")
    assert settings["chroma_auth_token_transport_header"] == "Authorization"


def test_parse_url_with_uri_token_auth() -> None:
    creds = "token"
    uri = (
        f"https://__auth_token__:{creds}@localhost:5432"
        "/mycollection?tenant=mytenant&database=mydb&batch_size=100&limit=10&offset=0"
    )
    parsed = CDPUri.from_uri(uri)
    assert parsed.host_or_path == "localhost"
    assert parsed.port == 5432
    assert parsed.database == "mydb"
    assert parsed.collection == "mycollection"
    assert parsed.tenant == "mytenant"
    assert parsed.batch_size == 100
    settings = _get_auth_settings_for_uri(parsed)
    assert settings is not None
    assert (
        settings["chroma_client_auth_provider"]
        == "chromadb.auth.token_authn.TokenAuthClientProvider"
    )
    assert settings["chroma_client_auth_credentials"] == creds
    assert settings["chroma_auth_token_transport_header"] == "Authorization"


def test_parse_url_with_env_xtoken_auth() -> None:
    os.environ["CHROMA_XTOKEN_AUTH"] = "token"
    uri = (
        "https://localhost:5432"
        "/mycollection?tenant=mytenant&database=mydb&batch_size=100&limit=10&offset=0"
    )
    parsed = CDPUri.from_uri(uri)
    assert parsed.host_or_path == "localhost"
    assert parsed.port == 5432
    assert parsed.database == "mydb"
    assert parsed.collection == "mycollection"
    assert parsed.tenant == "mytenant"
    assert parsed.batch_size == 100
    settings = _get_auth_settings_for_uri(parsed)
    assert settings is not None
    assert (
        settings["chroma_client_auth_provider"]
        == "chromadb.auth.token_authn.TokenAuthClientProvider"
    )
    assert settings["chroma_client_auth_credentials"] == os.getenv("CHROMA_XTOKEN_AUTH")
    assert settings["chroma_auth_token_transport_header"] == "X-Chroma-Token"


def test_parse_url_with_uri_xtoken_auth() -> None:
    creds = "token"
    uri = (
        f"https://__x_chroma_token__:{creds}@localhost:5432"
        "/mycollection?tenant=mytenant&database=mydb&batch_size=100&limit=10&offset=0"
    )
    parsed = CDPUri.from_uri(uri)
    assert parsed.host_or_path == "localhost"
    assert parsed.port == 5432
    assert parsed.database == "mydb"
    assert parsed.collection == "mycollection"
    assert parsed.tenant == "mytenant"
    assert parsed.batch_size == 100
    settings = _get_auth_settings_for_uri(parsed)
    assert settings is not None
    assert (
        settings["chroma_client_auth_provider"]
        == "chromadb.auth.token_authn.TokenAuthClientProvider"
    )
    assert settings["chroma_client_auth_credentials"] == creds
    assert settings["chroma_auth_token_transport_header"] == "X-Chroma-Token"
