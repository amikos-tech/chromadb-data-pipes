# Embedding Processors

## Default

CDP comes with a default embedding processor that supports the following embedding functions:

- Default (`default`) - The default ChromaDB embedding function based on OnnxRuntime and MiniLM-L6-v2 model.
- OpenAI (`openai`) - OpenAI's text-embedding-ada-002 model.

### Usage

#### Default

The below command will read a PDF files at the specified path, filter the output for a particular pdf (`grep`). Select
the first document's page, chunk it to 500 characters, embed each chunk using Chroma's default (MiniLM-L2-v2) model. The
resulting documents with embeddings will be written to `chroma-data.jsonl` file.

```bash
cdp imp pdf sample-data/papers/ | cdp tx chunk -s 500 | cdp tx embed --ef default > chroma-data.jsonl
```

#### OpenAI

The below command will read a PDF files at the specified path, filter the output for a particular pdf (`grep`). Select
the first document's page, chunk it to 500 characters, embed each chunk using OpenAI's text-embedding-ada-002 model.

```bash
cdp imp pdf sample-data/papers/ |grep "2401.02412.pdf" | head -1 | cdp tx chunk -s 500 | cdp tx embed --ef openai
```
