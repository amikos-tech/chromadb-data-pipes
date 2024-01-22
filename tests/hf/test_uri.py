from chroma_dp.huggingface import HFImportUri


def test_uri() -> None:
    uri = HFImportUri.from_uri(
        "hf://repo/name?limit=1000&offset=1000&split=train&stream=false&id_feature=_id&doc_feature=text&meta_features=title&embed_feature=openai"
    )

    assert uri.dataset == "repo/name"
    assert uri.limit == 1000
    assert uri.offset == 1000
    assert uri.split == "train"
    assert not uri.stream
    assert uri.id_feature == "_id"
    assert uri.doc_feature == "text"
    assert uri.meta_features == ["title"]

