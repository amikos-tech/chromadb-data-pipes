from typing import Dict, Any, Iterable, Optional, Annotated, List
import csv
import typer
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

from chroma_dp import CdpProducer, EmbeddableTextResource
from chroma_dp.processor.langchain_utils import convert_lc_doc_to_chroma_resource


class LangchainCSVProducer(CdpProducer[EmbeddableTextResource]):
    def __init__(
        self,
        path: str,
        document_column: Optional[str] = None,
        delimiter: Optional[str] = ",",
        quotechar: Optional[str] = '"',
        metadata_columns: Optional[List[str]] = None,
        batch_size: Optional[int] = 100,
    ) -> None:
        self.path = path
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.document_column = document_column
        self.metadata_columns = metadata_columns
        self.batch_size = batch_size or 100
        self._csv_args = {
            "delimiter": self.delimiter,
            "quotechar": self.quotechar,
        }
        self._all_columns: Optional[List[str]] = []
        with open(self.path) as f:
            csv_reader = csv.DictReader(f, **self._csv_args)  # type: ignore
            self._all_columns = csv_reader.fieldnames
            if (
                document_column is not None
                and document_column not in csv_reader.fieldnames
            ):
                raise ValueError(
                    f"Document column {document_column} not found in CSV file."
                )
            for mc in metadata_columns or []:
                if mc not in csv_reader.fieldnames:
                    raise ValueError(f"Metadata column {mc} not found in CSV file.")
            if not document_column:
                self._all_columns = self.metadata_columns

    def produce(
        self, limit: int = -1, offset: int = 0, **kwargs: Dict[str, Any]
    ) -> Iterable[EmbeddableTextResource]:
        loader = CSVLoader(
            file_path=self.path,
            metadata_columns=self._all_columns,
            csv_args={"delimiter": self.delimiter, "quotechar": self.quotechar},
        )

        docs = loader.load()
        _start = 0
        for _offset in range(_start, len(docs), self.batch_size):
            for doc in docs[_offset : _offset + self.batch_size]:
                doc_content = doc.page_content
                if self.document_column is not None:
                    doc_content = doc.metadata[self.document_column]
                    del doc.metadata[self.document_column]
                d = Document(page_content=doc_content, metadata=doc.metadata)
                yield convert_lc_doc_to_chroma_resource(d)


def csv_import(
    path: Annotated[
        str,
        typer.Argument(
            ...,
            help="The path to the directory containing the text files.",
        ),
    ],
    doc_feature: Annotated[
        Optional[str], typer.Option(help="The document feature.")
    ] = None,
    meta_features: Annotated[
        Optional[List[str]], typer.Option(help="The metadata features.")
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option(
            ...,
            "--batch-size",
            "-b",
            help="The batch size to use when processing the text files.",
        ),
    ] = 100,
    delimiter: Annotated[
        Optional[str],
        typer.Option(
            ...,
            "--delimiter",
            "-s",
            help="The delimiter to use when parsing the CSV file.",
        ),
    ] = ",",
    quotechar: Annotated[
        Optional[str],
        typer.Option(
            ...,
            "--quotechar",
            "-q",
            help="The quotechar to use when parsing the CSV file.",
        ),
    ] = '"',
) -> None:
    """Export text files from a directory to ChromaDB."""
    producer = LangchainCSVProducer(
        path=path,
        document_column=doc_feature,
        metadata_columns=meta_features,
        batch_size=batch_size,
        delimiter=delimiter,
        quotechar=quotechar,
    )
    for doc in producer.produce():
        typer.echo(doc.model_dump_json())
