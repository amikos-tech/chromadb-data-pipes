import json
import uuid
from typing import Any, Dict, Optional, Union

from langchain_core.documents import Document

from chroma_dp import EmbeddableTextResource, Metadata


def normalize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if metadata is None:
        return {}
    normalized_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, (str, float, bool, int)):
            normalized_metadata[k] = v
        elif isinstance(v, list):
            normalized_metadata[k] = ",".join(v)
        elif isinstance(v, dict):
            normalized_metadata[k] = json.dumps(v)
        else:
            normalized_metadata[k] = str(v)
    return normalized_metadata


def convert_chroma_emb_resource_to_lc_doc(doc: EmbeddableTextResource) -> Document:
    return Document(page_content=doc.text_chunk, metadata=doc.metadata)


def convert_lc_doc_to_chroma_resource(
    doc: Document,
    extra_metadata: Optional[Union[Dict[str, Any], Dict[str, Metadata]]] = None,
) -> EmbeddableTextResource:
    return EmbeddableTextResource(
        text_chunk=doc.page_content,
        embedding=None,
        # extra overrides doc.metadata
        metadata={
            **normalize_metadata(doc.metadata),
            **normalize_metadata(extra_metadata),
        },
        id=str(uuid.uuid4()),
    )
