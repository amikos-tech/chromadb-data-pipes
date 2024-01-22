import chromadb

from chroma_dp.huggingface import HFImportRequest, HFChromaDocumentSourceGenerator


def test_gen_streaming() -> None:
    client = chromadb.HttpClient()
    import_request = HFImportRequest(
        client=client,
        collection="test",
        dataset="tazarov/chroma-qna",
        create_collection=True,
        limit=10,
        offset=0,
        dataset_split="train",
        dataset_stream=True,
        document_feature="document",
        id_feature="id",
        embedding_feature="embedding",
        metadata_features=["metadata.title"],
        upsert=True,
    )
    gen = HFChromaDocumentSourceGenerator(import_request)

    count = 0
    for doc in gen:
        count += 1

    assert count == 10


def test_gen_non_streaming() -> None:
    client = chromadb.HttpClient()
    import_request = HFImportRequest(
        client=client,
        collection="test",
        dataset="tazarov/chroma-qna",
        create_collection=True,
        limit=10,
        offset=0,
        dataset_split="train",
        dataset_stream=True,
        document_feature="document",
        id_feature="id",
        embedding_feature="embedding",
        metadata_features=["metadata.title"],
        upsert=True,
    )
    gen = HFChromaDocumentSourceGenerator(import_request)

    count = 0
    for doc in gen:
        count += 1

    assert count == 10
