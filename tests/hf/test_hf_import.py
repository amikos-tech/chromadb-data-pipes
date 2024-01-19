import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from chroma_dp.huggingface import HFImportRequest


def test_hf_import() -> None:
    client = chromadb.HttpClient()
    _embedding_function = ONNXMiniLM_L6_V2()
    import_request = HFImportRequest(
        client=client,
        collection="test",
        dataset="KShivendu/dbpedia-entities-openai-1M",
        create_collection=True,
        limit=1000,
        dataset_split="train",
        dataset_stream=False,
        embedding_function=_embedding_function,
        document_feature="text",
        id_feature="_id",
        embedding_feature="openai",
        upsert=True,
    )
    # TODO add test

    # run(import_request)
