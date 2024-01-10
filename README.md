# Chroma Data Pump

> WARNING: This is still work in progress

CLI for managing data in Chroma DB.

## Installation

```bash
pip install chromadb-dp
```

## Usage

Help:

```bash
cdp --help
```

### Import

> Note: For importing private dataset to HF make sure that you have your HF token as env
> var - `export HF_TOKEN=hf_abcd1234`

Common import options:

| Option               | Type    | Description                                                           |
|----------------------|---------|-----------------------------------------------------------------------|
| --doc-feature        | TEXT    | The document feature. Default: `None`                                 |
| --chroma-endpoint    | TEXT    | The Chroma endpoint. Default: `None`                                  |
| --collection         | TEXT    | The Chroma collection. Default: `None`                                |
| --create             | FLAG    | Create the Chroma collection if it does not exist. Default: disabled  |
| --embed-feature      | TEXT    | The embedding feature. Default: `None`                                |
| --meta-features      | TEXT    | The metadata features. Default: `None`                                |
| --id-feature         | TEXT    | The id feature. Default: `None`                                       |
| --limit              | INTEGER | The limit. Default: `-1` (the whole dataset will be imported)         |
| --offset             | INTEGER | The offset. Default: `0` (the dataset is imported from the beginning) |
| --batch-size         | INTEGER | The batch size. Default: `100`                                        |
| --embedding-function | default | The embedding function. Default: `default`                            |
| --upsert             | FLAG    | Upsert documents. Default: disabled                                   |
| --cdp-uri            | TEXT    | The ChromaDP URI. Default: `None`                                     |

#### HuggingFace Import

| Option    | Type | Description                                                                      |
|-----------|------|----------------------------------------------------------------------------------|
| --dataset | TEXT | The HuggingFace dataset. Expected format: `<user>/<dataset_id>`. Default: `None` |
| --split   | TEXT | The Hugging Face. Default: `train`                                               |
| --stream  | FLAG | Stream dataset instead of downloading. Default: disabled                         |
| --hf-uri  | TEXT | The Hugging Face URI. Default: `None`                                            |

```bash
cdp hf import \
  --dataset KShivendu/dbpedia-entities-openai-1M \
  --collection chroma-qna \
  --create \
  --chroma-endpoint http://localhost:8000 \
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
  --hf-uri "hf:KShivendu/dbpedia-entities-openai-1M?split=train" \
  --offset 0 \
  --limit 1000 \
  --id-feature _id \
  --doc-feature text \
  --meta-features title \
  --embed-feature openai \
  --cdp-uri "http://localhost:8000/default_database/test?create_collection=true"
```

URI only import:

```bash
cdp hf import \
  --hf-uri "hf:KShivendu/dbpedia-entities-openai-1M?limit=1000&offset=1000&split=train&stream=false&id_feature=_id&doc_feature=text&meta_features=title&embed_feature=openai" \
  --cdp-uri "http://localhost:8000/default_database/test?create_collection=true"
```

### Export

Common export options:

| Option            | Type    | Description                          |
|-------------------|---------|--------------------------------------|
| --chroma-endpoint | TEXT    | The Chroma endpoint. Default: None   |
| --collection      | TEXT    | The Chroma collection. Default: None |
| --meta-features   | TEXT    | The metadata features. Default: None |
| --limit           | INTEGER | The limit. Default: -1               |
| --offset          | INTEGER | The offset. Default: 0               |
| --batch-size      | INTEGER | The batch size. Default: 100         |
| --cdp-uri         | TEXT    | The ChromaDP URI. Default: None      |
| --out             | TEXT    | The output path. Default: None       |

#### HuggingFace Export

> Note: For exporting dataset to HF make sure that you have your HF token as env var - `export HF_TOKEN=hf_abcd1234`

| Option    | Type | Description                             |
|-----------|------|-----------------------------------------|
| --dataset | TEXT | The Hugging Face dataset. Default: None |
| --split   | TEXT | The Hugging Face. Default: train        |
| --upload  | FLAG | Upload. Default: no-upload              |
| --private | FLAG | Private HF Repo. Default: no-private    |
| --hf-uri  | TEXT | The Hugging Face URI. Default: None     |

```bash
cdp hf export \
  --dataset tazarov/chroma-qna \
  --collection chroma-qna \
  --chroma-endpoint http://localhost:8000 \
  --limit 1000 \
  --meta-features title \
  --upload
```

Exporting with URI and mix of parameters:

```bash
cdp hf export \
  --hf-uri hf:tazarov/dataset123-1 \
  --cdp-uri http://localhost:8000/default_database/test \
  --upload \
  --private
```

Exporting with URI only:

```bash
cdp hf export \
  --hf-uri "hf:tazarov/dataset123-1?split=train&meta_features=title" \
  --cdp-uri "http://localhost:8000/default_database/test?limit=1000&offset=1000" \
  --upload \
  --private
```
