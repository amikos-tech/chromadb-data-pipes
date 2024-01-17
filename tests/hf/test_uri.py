from chroma_dp.huggingface import HFImportUri


def test_uri():
    uri = HFImportUri.from_uri(
        "hf:repo/name?limit=1000&offset=1000&split=train&stream=false&id_feature=_id&document_feature=text&meta_features=title&embed_feature=openai"
    )

    print(uri)
