from chroma_dp.utils.chroma import CDPUri


def test_parse_cdp_uri_basic():
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


def test_parse_cdp_uri_local_absolute():
    uri = "file:///abs/path/persist_dir/some_collection?tenant=mytenant&database=db1&batch_size=100&limit=10&offset=0"
    parsed = CDPUri.from_uri(uri)
    assert parsed.is_local is True
    assert parsed.host_or_path == "/abs/path/persist_dir/"
    assert parsed.tenant == "mytenant"
    assert parsed.database == "db1"
    assert parsed.collection == "some_collection"


def test_parse_cdp_uri_local_relative():
    uri = "file://abs/path/persist_dir/some_collection?tenant=mytenant&database=db1&batch_size=100&limit=10&offset=0"
    parsed = CDPUri.from_uri(uri)
    assert parsed.is_local is True
    assert parsed.host_or_path == "abs/path/persist_dir/"
    assert parsed.tenant == "mytenant"
    assert parsed.database == "db1"
    assert parsed.collection == "some_collection"
