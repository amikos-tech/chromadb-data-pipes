# Embedding Processors

## Default Embedding Processor

CDP comes with a default embedding processor that supports the following embedding functions:

- Default (`default`) - The default ChromaDB embedding function based on OnnxRuntime and MiniLM-L6-v2 model.
- OpenAI (`openai`) - OpenAI's text-embedding-ada-002 model.
- Cohere (`cohere`) - Cohere's embedding models.
- HuggingFrace (`hf`) - HuggingFace's embedding models.
- SentenceTransformers (`st`) - SentenceTransformers' embedding models.

The embedding functions are based on [ChromaDB's embedding functions](https://docs.trychroma.com/embeddings).

### Usage

#### Default

The below command will read a PDF files at the specified path, filter the output for a particular pdf (`grep`). Select
the first document's page, chunk it to 500 characters, embed each chunk using Chroma's default (MiniLM-L2-v2) model. The
resulting documents with embeddings will be written to `chroma-data.jsonl` file.

```bash
cdp imp pdf sample-data/papers/ | cdp chunk -s 500 | cdp embed --ef default > chroma-data.jsonl
```

#### OpenAI

To use this embedding function, you need to install the `openai` python package.

```bash
pip install openai
```

!!! note "OpenAI API Key"

    You need to have an OpenAI API key to use this embedding function. 
    You can get an API key by signing up for an account at [OpenAI API Keys page](https://platform.openai.com/api-keys).
    The API key must be exported as env variable `OPENAI_API_KEY=sk-xxxxxx`.

!!! note "OpenAI Embedding Models"

    By default, if not specified, the `text-embedding-ada-002` model is used.
    You can pass in an optional `--model=text-embedding-3-small` argument or
    env variable `GEMINI_MODEL_NAME=text-embedding-3-large` , which lets you choose which OpenAI embeddings model to use.

The below command will read a PDF files at the specified path, filter the output for a particular pdf (`grep`). Select
the first document's page, chunk it to 500 characters, embed each chunk using OpenAI's text-embedding-ada-002 model.

```bash
export OPENAI_API_KEY=sk-xxxxxx
cdp imp pdf sample-data/papers/ |grep "2401.02412.pdf" | head -1 | cdp chunk -s 500 | cdp embed --ef openai
```

#### Cohere

To use this embedding function, you need to install the `cohere` python package.

```bash
pip install cohere
```

!!! note "Cohere API Key"

    You need to have a Cohere API key to use this embedding function. You can get an API key by signing up for an account
    at [Cohere](https://dashboard.cohere.ai/welcome/register).
    The API key must be exported as env variable `COHERE_API_KEY=x4q...`.

!!! note "Cohere Embedding Models"

    By default, if not specified, the `embed-english-v3.0` model is used.
    You can pass in an optional `--model=embed-english-light-v3.0` argument or
    env variable `COHERE_MODEL_NAME=embed-multilingual-v3.0` , which lets you choose which Cohere embeddings model to use.
    More about available models can be found at [Cohere's API docs](https://docs.cohere.com/reference/embed)

The below command will read a PDF files at the specified path, select
the last document's page, chunk it to 100 characters, embed each chunk using Cohere's embed-english-light-v3.0 model.

```bash
export COHERE_API_KEY=x4q
export COHERE_MODEL_NAME="embed-english-light-v3.0"
cdp imp pdf sample-data/papers/ | tail -1 | cdp chunk -s 100 | cdp embed --ef cohere
```

#### HuggingFace

!!! note "HF API Token"

    You need to have a HuggungFace API token to use this embedding function.
    Create or use one from your [tokens page](https://huggingface.co/settings/tokens).
    The API key must be exported as env variable `HF_TOKEN=hf_xxxx`.

!!! note "HF Embedding Models"

    By default, if not specified, the `sentence-transformers/all-MiniLM-L6-v2` model is used.
    You can pass in an optional `--model=BAAI/bge-large-en-v1.5` argument or
    env variable `HF_MODEL_NAME=BAAI/bge-large-en-v1.5` , which lets you choose which 
    Hugging Frace embeddings model to use.

The below command will read a PDF files at the specified path, select
the first two pages, chunk it to 150 characters, selects the last chunk and embeds the chunk
using BAAI/bge-large-en-v1.5 model.

```bash
export HF_TOKEN=hf_xxxx
export HF_MODEL_NAME="BAAI/bge-large-en-v1.5"
cdp imp pdf sample-data/papers/ | head -2 | cdp chunk -s 150 | tail -1 | cdp embed --ef hf
```

#### SentenceTransformers

To use this embedding function, you need to install the `sentence-transformers` python package.

```bash
pip install sentence-transformers
```

!!! note "SentenceTransformers Embedding Models"

    By default, if not specified, the `all-MiniLM-L6-v2` model is used.
    You can pass in an optional `--model=BAAI/bge-large-en-v1.5` argument or
    env variable `ST_MODEL_NAME=BAAI/bge-large-en-v1.5` , which lets you choose which Sentence Transformers
    embeddings model to use.

The below command will read a PDF files at the specified path, select
the first two pages, chunk it to 150 characters, selects the last chunk and embeds the chunk
using BAAI/bge-small-en-v1.5 model.

```bash
export ST_MODEL_NAME="BAAI/bge-small-en-v1.5"
cdp imp pdf sample-data/papers/ | head -2 | cdp chunk -s 150 | tail -1 | cdp embed --ef st
```

#### Google Generative AI Embedding (Gemini)

To use Google Generative AI Embedding (Gemini) function, you need to install the `google-generativeai` python package.

```bash
pip install google-generativeai
```

!!! note "Google API Key"

    You need to have a Google API key to use this embedding function.
    To manage your keys go to [Maker Suite](https://makersuite.google.com/).
    The API key must be exported as env variable `GEMINI_API_KEY=xxxx`.

!!! note "Models"

    By default, if not specified, the `models/embedding-001` model is used.
    You can pass in an optional `--model=models/embedding-001` argument or
    env variable `GEMINI_MODEL_NAME=models/embedding-001`, which lets you choose which Gemini embeddings model to use.

!!! note "Task Type"

    The embedding function also supports task type parameter. By default we use `RETRIEVAL_DOCUMENT`, For more details
    visit [Gemini API Docs](https://ai.google.dev/examples/doc_search_emb#api_changes_to_embeddings_with_model_embedding-001).

The below command will read a PDF files at the specified path, select the first two pages, chunk it to 150 characters, 
selects the last chunk and embeds the chunk using `models/embedding-001` model.

```bash
cdp imp pdf sample-data/papers/ | head -2 | cdp chunk -s 150 | tail -1 | cdp embed --ef gemini
```
