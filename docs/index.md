# ChromaDB Data Pipes 🖇️| Rediscover AI/ML the Unix Way

ChromaDB Data Pipes is a collection of tools to build data pipelines for Chroma DB, inspired by the Unix philosophy of "
do one thing and do it well".

## Installation

```bash
pip install chromadb-data-pipes
```

## Usage

**Get help:**

```bash
cdp --help
```

### Importing

**Import data from HuggingFace Datasets to `.jsonl` file:**

```bash
cdp imp hf --uri "hf:tazarov/chroma-qna?split=train" > chroma-qna.jsonl
```

**Import data from HuggingFace Datasets to Chroma DB:**

The below command will import the `train` split of the given dataset to Chroma chroma-qna `chroma-qna` collection. The
collection will be created if it does not exist and documents will be upserted.

```bash
cdp imp hf --uri "hf://tazarov/chroma-qna?split=train" | cdp imp chroma --uri "http://localhost:8000/default_database/chroma-qna" --upsert --create
```

### Exporting

**Export data from Chroma DB to `.jsonl` file:**

The below command will export the first 10 documents from the `chroma-qna` collection to `chroma-qna.jsonl` file.

```bash
cdp exp chroma --uri "http://localhost:8000/default_database/chroma-qna" --limit 10 > chroma-qna.jsonl
```

**Export data from Chroma DB to HuggingFace Datasets:**

The below command will export the first 10 documents with offset 10 from the `chroma-qna` collection to HuggingFace
Datasets `tazarov/chroma-qna` dataset. The dataset will be uploaded to HF.

!!! note HF Auth and Privacy

    Make sure you have `HF_TOKEN=hf_....` environment variable set.
    If you want your dataset to be private, add `--private` flag to the `cdp exp hf` command.

```bash
cdp exp chroma --uri "http://localhost:8000/default_database/chroma-qna" --limit 10 --offset 10 | cdp exp hf --uri "hf://tazarov/chroma-qna-modified"
```

To export a dataset to a file, use `--uri` with `file://` prefix:

```bash
cdp exp chroma --uri "http://localhost:8000/default_database/chroma-qna" --limit 10 --offset 10 | cdp exp hf --uri "file://chroma-qna"
```

!!! note File Location

    The file is  relative to the current working directory.

### Transforming

**Copy collection from one Chroma collection to another and re-embed the documents:**

```bash
cdp exp chroma --uri "http://localhost:8000/default_database/chroma-qna" | cdp tx embed --ef default | cdp imp chroma --uri "http://localhost:8000/default_database/chroma-qna-def-emb" --upsert --create
```

**Import dataset from HF to Chroma and embed the documents:**

```bash
cdp imp hf --uri "hf://tazarov/ds2?split=train" | cdp tx embed --ef default | cdp imp chroma --uri "http://localhost:8000/default_database/chroma-qna-def-emb-hf" --upsert --create
```

### Misc

**Count the number of documents in a collection:**

```bash
cdp exp chroma --uri "http://localhost:8000/default_database/chroma-qna" | wc -l
```
