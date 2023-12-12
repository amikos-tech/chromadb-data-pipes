# About

Small Python utility to import HF datasets to Chroma DB as well as export Chroma DB collections to HF datasets.

## Installation

```bash
pip install chromadb-hfds
```

## Usage

### Import

- dataset (e.g. `KShivendu/dbpedia-entities-openai-1M`)
- collection name (e.g. `dbpedia-entities-openai-1M`)
- chroma_host (e.g. `http://localhost:8080`)
- document_column (e.g. `text`)
- embedding_column (e.g. `openai`)
- metadata_columns (e.g. `["label", "uri"]`)
- id_column (e.g. `uri` , defaults to `None` which will trigger automatic id generation)
- limit (e.g. `1000`)
- offset (e.g. `0`)
- batch_size (e.g. `100`)
- embedding_function (chromadb embedding function, e.g. `openai`)
- create_collection (e.g. `True`)

```python
from chromadb_hfds import chromadb_hfds_import

chromadb_hfds_import.from_dataset('').
    into(host="http://localhost:8000", collection='collection_name', create=True, ).
    with_document_column('text').
    with_embedding_column('openai').
    with_metadata_columns(['label', 'uri']).
    with_id_column('uri').
    with_limit(1000).
    with_offset(0).
    with_batch_size(100).
    with_embedding_function('openai').
    run()
```

Commandline:

```bash

pythom -m chromadb_hdfs.import \
  --dataset KShivendu/dbpedia-entities-openai-1M \
  --collection dbpedia-entities-openai-1M \
  --host http://localhost:8000 --document_column text \
  --embedding_column openai --metadata_columns label,uri \
  --id_column uri \
  --limit 1000 \
  --offset 0 \
  --batch_size 100 \
  --embedding_function openai \
  --create_collection True
```