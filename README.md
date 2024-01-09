# Chroma Data Pump

> WARNING: This is still work in progress

Python library for your data management for Chroma DB.

## Installation

```bash
pip install chromadb-dp
```

## Usage

```bash
chroma-dp --help # get help
chroma-dp hf import \
  --dataset tazarov/chroma-qna \
  --collection chroma-qna \
  --create \
  --chroma-endpoint http://localhost:8080 \
  --limit 1000 \
  --doc-feature text \
  --embed-feature openai \
  --meta-features title \
  --id-feature _id \
  --upsert \
  --batch-size 100
```
```bash
cdp hf import \
  --hf-uri "hf:KShivendu/dbpedia-entities-openai-1M" \
  --offset 0 \
  --limit 1000 \
  --split train \
  --id-feature _id \
  --doc-feature text \
  --meta-features title \
  --embed-feature openai \
  --cdp-uri "http://localhost:8000/_/test?create_collection=true"
```

```bash
cdp hf import --hf-uri "hf:KShivendu/dbpedia-entities-openai-1M?limit=1000&offset=1000&split=train&stream=false&id_feature=_id&doc_feature=text&meta_features=title&embed_feature=openai" \
  --cdp-uri "http://localhost:8000/_/test?create_collection=true"

```

```bash
cdp hf export \
  --dataset tazarov/chroma-qna \
  --collection chroma-qna \
  --chroma-endpoint http://localhost:8080 \
  --limit 1000 \
  --metadata-features title \
  --upload
```

### Import

- dataset (e.g. `KShivendu/dbpedia-entities-openai-1M`) - the id of the HuggingFace dataset to import.
- collection name (e.g. `dbpedia-entities-openai-1M`) - The name of the collection to import into.
- create (e.g. `True`) - create collection if it doesn't exist
- chroma endpoint (e.g. `http://localhost:8080`) - the endpoint of the Chroma DB instance to import into.
- document feature (e.g. `text`) - The dataset feature to use as the document text.
- embedding feature (e.g. `openai`) - The dataset feature to use as the document embedding. If not provided, the
  embedding will be generated automatically.
- metadata features (e.g. `["label", "uri"]`) - The dataset features to use as document metadatas.
- id feature (e.g. `_id` ) - The dataset feature to use as the document id. If not provided, the id will be generated
  automatically.
- limit (e.g. `1000`) - The number of documents to import.
- offset (e.g. `0`) - The offset to start importing from. (Not yet implemented)
- batch_size (e.g. `100`) - The batch size to use for importing.
- embedding_function (chromadb embedding function, e.g. `default`) - The embedding function to use for generating
  embeddings. If not provided, the default embedding function will be used.

```python
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from chroma_dp.chroma_import import ChromaDBImportBuilder, ImportRequest

import_req = ImportRequest(
    dataset="KShivendu/dbpedia-entities-openai-1M",
    dataset_split="train",
    dataset_stream=False,
    endpoint="http://localhost:8000",
    collection="dbpedia",
    create=True,
    document_feature="text",
    embedding_feature="openai",
    metadata_features=["title"],
    id_feature="_id",
    limit=10000,
    offset=0,
    batch_size=100,
    embedding_function=DefaultEmbeddingFunction(),
    client=None,
)

ChromaDBImportBuilder(import_req).run()
```

Commandline:

```bash
cdp \
    --dataset-id "KShivendu/dbpedia-entities-openai-1M" \
    --collection test123 --create \
    --chroma-endpoint "http://localhost:8000" \
    --limit 1000 \
    --document-feature text \
    --embedding-feature openai \
    --metadata-features title \
    --id-feature _id
```
