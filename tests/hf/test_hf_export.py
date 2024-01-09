import chromadb

from chroma_dp.huggingface.hf_export import HFExportRequest, run


def test_hf_import():
    client = chromadb.HttpClient()
    export_request = HFExportRequest(
        client=client,
        collection="test",
        dataset="tazarov/test",
        dataset_file="test/hf",
        upload=True,
    )

    rest = run(export_request)
    print(rest)
