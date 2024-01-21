# ChromaDB Data Pipes 🖇️ - The easiest way to get data into and out of ChromaDB

ChromaDB Data Pipes is a collection of tools to build data pipelines for Chroma DB, inspired by the Unix philosophy of
"do one thing and do it well".

Roadmap:

- ✅ Integration with LangChain 🦜🔗
- 🚫 Integration with LlamaIndex 🦙
- ✅ Support more than `all-MiniLM-L6-v2` as embedding functions (head over
  to [Embedding Processors](https://datapipes.chromadb.dev/processors/embedding/) for more info)
- 🚫 Multimodal support
- ♾️ Much more!

## Installation

```bash
pip install chromadb-data-pipes
```

## Usage

**Get help:**

```bash
cdp --help
```

### Example Use Cases

This is a short list of use cases to evaluate whether this is the right tool for your needs:

- Importing large datasets from local documents (PDF, TXT, etc.), from HuggingFace, from local persisted Chroma DB or
  even another remote Chroma DB.
- Exporting large dataset to HuggingFace or any other dataformat supported by the library (if your format is not
  supported, either implement it in a small function or open an issue)
- Create a dataset from your data that you can share with others (including the embeddings)
- Clone Collection with different embedding function, distance function, and other HNSW fine-tuning parameters
- Re-embed documents in a collection with a different embedding function
- Backup your data to a `jsonl` file
- Use other existing unix or other tools to transform your data after exporting from or before importing into Chroma DB

### Importing

**Import data from HuggingFace Datasets to `.jsonl` file:**

```bash
cdp ds-get "hf://tazarov/chroma-qna?split=train" > chroma-qna.jsonl
```

**Import data from HuggingFace Datasets to Chroma DB:**

The below command will import the `train` split of the given dataset to Chroma chroma-qna `chroma-qna` collection. The
collection will be created if it does not exist and documents will be upserted.

```bash
cdp ds-get "hf://tazarov/chroma-qna?split=train" | cdp import "http://localhost:8000/chroma-qna" --upsert --create
```

**Importing from a directory with PDF files into Local Persisted Chroma DB:**

```bash
cdp imp pdf sample-data/papers/ | grep "2401.02412.pdf" | head -1 | cdp chunk -s 500 | cdp embed --ef default | cdp import "file://chroma-data/my-pdfs" --upsert --create
```

> Note: The above command will import the first PDF file from the `sample-data/papers/` directory, chunk it into 500
> word chunks, embed each chunk and import the chunks to the `my-pdfs` collection in Chroma DB.

### Exporting

**Export data from Local Persisted Chroma DB to `.jsonl` file:**

The below command will export the first 10 documents from the `chroma-qna` collection to `chroma-qna.jsonl` file.

```bash
cdp export "file://chroma-data/chroma-qna" --limit 10 > chroma-qna.jsonl
```

**Export data from Local Persisted Chroma DB to `.jsonl` file with filter:**

The below command will export data from local persisted Chroma DB to a `.jsonl` file using a `where` filter to select
the documents to export.

```bash
cdp export "file://chroma-data/chroma-qna" --where '{"document_id": "123"}' > chroma-qna.jsonl
```

**Export data from Chroma DB to HuggingFace Datasets:**

The below command will export the first 10 documents with offset 10 from the `chroma-qna` collection to HuggingFace
Datasets `tazarov/chroma-qna` dataset. The dataset will be uploaded to HF.

> HF Auth and Privacy: Make sure you have `HF_TOKEN=hf_....` environment variable set. If you want your dataset to
> be private, add `--private` flag to the `cdp ds-put` command.

```bash
cdp export "http://localhost:8000/chroma-qna" --limit 10 --offset 10 | cdp ds-put "hf://tazarov/chroma-qna-modified"
```

To export a dataset to a file, use `--uri` with `file://` prefix:

```bash
cdp export "http://localhost:8000/chroma-qna" --limit 10 --offset 10 | cdp ds-put "file://chroma-qna"
```

> File Location The file is relative to the current working directory.

### Processing

**Copy collection from one Chroma collection to another and re-embed the documents:**

```bash
cdp export "http://localhost:8000/chroma-qna" | cdp embed --ef default | cdp import "http://localhost:8000/chroma-qna-def-emb" --upsert --create
```

> Note: See [Embedding Processors](./processors/embedding.md) for more info about supported embedding functions.

**Import dataset from HF to Local Persisted Chroma and embed the documents:**

```bash
cdp ds-get "hf://tazarov/ds2?split=train" | cdp embed --ef default | cdp import "file://chroma-data/chroma-qna-def-emb-hf" --upsert --create
```

**Chunk Large Documents:**

```bash
cdp imp pdf sample-data/papers/ | grep "2401.02412.pdf" | head -1 | cdp chunk -s 500
```

### Misc

**Count the number of documents in a collection:**

```bash
cdp export "http://localhost:8000/chroma-qna" | wc -l
```
