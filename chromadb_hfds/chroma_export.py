from typing import TypedDict, Optional


class ExportRequest(TypedDict):
    dataset: str
    dataset_split: str
    endpoint: str
    collection: str
    limit: Optional[int]
    offset: Optional[int]
    batch_size: Optional[int]
    upload: Optional[bool]


class ExportResult(TypedDict):
    pass


class ChromaDBExporter(object):
    def __init__(self, export_request: ExportRequest) -> None:
        pass

    def export(self) -> ExportResult:

        pass

class ChromaDBExportBuilder(object):
    def __init__(self, export_request: Optional[ExportRequest] = None) -> None:
        self._dataset = export_request.get("dataset") if export_request else None
        self._chroma_endpoint = (
            export_request.get("endpoint") if export_request else None
        )
        self._collection = export_request.get("collection") if export_request else None
        self._dataset_split = (
            export_request.get("dataset_split") if export_request else "train"
        )
        self._limit = export_request.get("limit") if export_request else -1
        self._offset = export_request.get("offset") if export_request else None
        self._batch_size = export_request.get("batch_size") if export_request else 100
        self._upload = export_request.get("upload") if export_request else False
        self._chroma_client = chromadb.HttpClient(host=self._chroma_endpoint)

    @classmethod
    def from_dataset(
        cls,
        dataset: str,
        dataset_split: str = "train",
        endpoint: str = "http://localhost:8000",
        collection: str = "default",
        limit: int = -1,
        offset: Optional[int] = None,
        batch_size: int = 100,
        upload: bool = False,
    ) -> "ChromaDBExportBuilder":
        return cls(
            export_request={
                "dataset": dataset,
                "dataset_split": dataset_split,
                "endpoint": endpoint,
                "collection": collection,
                "limit": limit,
                "offset": offset,
                "batch_size": batch_size,
                "upload": upload,
            }
        )

    def build(self) -> ChromaDBExporter:
        return ChromaDBExporter(
            export_request={
                "dataset": self._dataset,
                "dataset_split": self._dataset_split,
                "endpoint": self._chroma_endpoint,
                "collection": self._collection,
                "limit": self._limit,
                "offset": self._offset,
                "batch_size": self._batch_size,
                "upload": self._upload,
            }
        )

