from chroma_dp.utils.chroma import CDPUri


def test_parse_cdp_uri_basic():
    uri = "https://basic_user:basic_password@localhost:5432" \
          "/mydb/mycollection?tenant=mytenant&batch_size=100&limit=10&offset=0"
    parsed = CDPUri.from_uri(uri)
    assert parsed.host == "localhost"
    assert parsed.port == 5432
    assert parsed.database == "mydb"
    assert parsed.collection == "mycollection"
    assert parsed.tenant == "mytenant"
    assert parsed.batch_size == 100
    assert parsed.limit == 10
    assert parsed.offset == 0
    assert parsed.auth is not None
    assert parsed.auth["type"] == "basic"
    assert parsed.auth["username"] == "basic_user"
    assert parsed.auth["password"] == "basic_password"
