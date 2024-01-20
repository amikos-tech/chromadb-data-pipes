# Chunking

## Usage

```bash
cdp imp url <url> [flags]
```

The following example will chunk the document into 500 character chunks and print the chunks to stdout. We will also add (`-a`
option) the offset position of each chunk within the document as metadata `start_index`.

```bash
cdp imp pdf sample-data/papers/ | grep "2401.02412.pdf" | head -1 | cdp chunk -s 500 -a
```

Alternatively you can chunk from an input `jsonl` file:

```bash
cdp imp pdf sample-data/papers/ | grep "2401.02412.pdf" | head -1 | cdp chunk -s 500 > chunk.jsonl
```

!!! warning "jsonl format"

    It is expected that the `jsonl` file contains `chroma_dp.EmbeddableTextResource` objects (one per line).

```bash
cdp chunk -s 500 --in chunk.jsonl
```

!!! note "Help"

    Run `cdp chunk --help` for more information.
