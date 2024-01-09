from typing import Optional

import datasets
from datasets import Dataset
from datasets.config import HF_ENDPOINT
from huggingface_hub import HfApi, DatasetCard
from pydantic import Field
from rich.progress import track

from chroma_dp import ExportResult, ExportRequest
from chroma_dp.huggingface.utils import _infer_hf_type
from chroma_dp.utils.chroma import check_collection_exists, get_collection, read_large_data_in_chunks


class HFExportRequest(ExportRequest):
    dataset: str = Field(..., description="Dataset name.")
    dataset_split: Optional[str] = Field("train", description="Dataset split.")
    dataset_file: Optional[str] = Field(..., description="Path to dataset file.")
    upload: Optional[bool] = Field(False, description="Upload to ChromaDB.")
    private: Optional[bool] = Field(False, description="Make dataset private on Hugging Face Hub.")


def run(export_request: HFExportRequest) -> ExportResult:
    _exists = check_collection_exists(export_request.client, export_request.collection)

    if not _exists:
        return ExportResult(success=False, message="Collection does not exist.")

    data = {
        "id": [],
        "embedding": [],
        "document": [],
        # metadata is added as `metadata.{key}`
    }
    metadata_feature = None
    _collection = get_collection(export_request.client, export_request.collection)
    col_count = _collection.count()
    total_results_to_fetch = min(col_count, export_request.limit) if export_request.limit > 0 else col_count
    export_batch_size = export_request.batch_size
    _start = 0
    for offset in track(range(_start, total_results_to_fetch, export_batch_size), description="Exporting Dataset"):
        result = read_large_data_in_chunks(_collection, offset=offset, limit=min(
            total_results_to_fetch - offset, export_batch_size))
        for i, id_ in enumerate(result["ids"]):
            data["id"].append(str(id_))
            data["embedding"].append(
                result["embeddings"][i] if result["embeddings"] else None)
            data["document"].append(
                result["documents"][i] if result["documents"] else None)
            # TODO we need to check if this works if items where metadata is None for a doc
            if result["metadatas"][i]:
                for key in result["metadatas"][i].keys():
                    if f"metadata.{key}" not in data:
                        data[f"metadata.{key}"] = []
                    data[f"metadata.{key}"].append(
                        result["metadatas"][i][key])
                if not metadata_feature:
                    metadata_feature = {}
                for key in result["metadatas"][i].keys():
                    if key not in metadata_feature:
                        metadata_feature[f"metadata.{key}"] = _infer_hf_type(
                            result["metadatas"][i][key])
    features = datasets.Features({
        "id": datasets.Value("string"),
        # TODO this should be int or float
        "embedding": datasets.features.Sequence(feature=datasets.Value(dtype='float32')),
        "document": datasets.Value("string"),
        **(metadata_feature if metadata_feature else {})
    })
    dataset = Dataset.from_dict(data, features=features, info=datasets.DatasetInfo(
        description=f"Chroma Dataset for collection {_collection.name}", features=features))
    dataset.save_to_disk(str(export_request.dataset_file))

    if export_request.upload and dataset:
        dataset.push_to_hub(export_request.dataset, private=export_request.private)
        # TODO add metadata to feature mapping in the x-chroma field
        custom_metadata = {
            "license": "mit",
            "language": "en",
            "pretty_name": f"Chroma export of collection {_collection.name}",
            "size_categories": ["n<1K"],
            "x-chroma": {
                "description": f"Chroma Dataset for collection {_collection.name}",
                "collection": _collection.name,
                "metadata": _collection.metadata,
            }}
        card = DatasetCard.load(
            repo_id_or_path=export_request.dataset,
            repo_type="dataset")
        data_info = card.data
        data_dict = {**data_info.to_dict(), **custom_metadata}
        card.content = f"---\n{str(data_dict)}\n---\n{card.text}"
        HfApi(endpoint=HF_ENDPOINT).upload_file(
            path_or_fileobj=str(card).encode(),
            path_in_repo="README.md",
            repo_id=export_request.dataset,
            repo_type="dataset",
        )

    #
    #
    # if len(data["id"]) > 0:
    #     features = datasets.Features({
    #         "id": datasets.Value("string"),
    #         # TODO this should be int or float
    #         "embedding": datasets.features.Sequence(
    #             feature=datasets.Value(dtype='float32')
    #         ),
    #         "document": datasets.Value("string"),
    #         **metadata_feature
    #     })
    #     dataset = Dataset.from_dict(data, features=features)
    #     dataset.save_to_disk(str(dataset_path) + f"-{offset}")
    #     temp_datasets.append(dataset)
    #
    # if len(temp_datasets) >= 1:
    #     dataset = concatenate_datasets(temp_datasets)
    #     dataset.save_to_disk(dataset_path)
    #     for safe_path in temp_dateset_paths:
    #         shutil.rmtree(safe_path)
    #     if dataset_file:
    #         dataset.save(dataset_file)
    #     _persist_export_state(_export_state_file, {
    #         **_export_config, "safe_point_paths": [], "finished": True})
    #     if dataset_upload:
    #         dataset.push_to_hub(dataset_remote_path)
    #         custom_metadata = {
    #             "license": "mit",
    #             "language": "en",
    #             "pretty_name": f"Chroma export of collection {col.name}",
    #             "size_categories": ["n<1K"],
    #             "x-chroma": {
    #                 "description": f"Chroma Dataset for collection {col.name}",
    #                 "collection": col.name,
    #                 "metadata": col.metadata,
    #             }}
    #         # TODO move this in a separate function
    #         card = DatasetCard.load(
    #             repo_id_or_path=dataset_remote_path,
    #             repo_type="dataset")
    #         data_info = card.data
    #         data_dict = {**data_info.to_dict(), **custom_metadata}
    #         card.content = f"---\n{str(data_dict)}\n---\n{card.text}"
    #         HfApi(endpoint=datasets.config.HF_ENDPOINT).upload_file(
    #             path_or_fileobj=str(card).encode(),
    #             path_in_repo="README.md",
    #             repo_id=dataset_remote_path,
    #             repo_type="dataset",
    #         )

    return ExportResult(success=True, message="Export successful.")
