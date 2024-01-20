# URL Importer

Imports data from a URL.

## Default

### Usage

```bash
cdp imp url <url> [flags]
```

!!! note "Get Help"

    Get help for the command with the following flag:

    ```bash
    cdp imp url --help
    ```

!!! warn "URLs with binary data"

    Currently the URL importer does not support URLs with binary data, such as PDFs or other binary files.
    Need this? Raise an issue [here](https://github.com/amikos-tech/chromadb-data-pipes/issues/new).

The following example imports data from ChromaDB documentation with max depth 2.

```bash
cdp imp url https://docs.trychroma.com/embeddings -d 2
```

#### Advanced Usage

The following example imports data from ChromaDB documentation with max depth 3,
chunks the data into 512 byte chunks, cleans the data of emojis, and embeds the data using the default embedding
function.

```bash
cdp imp url https://docs.trychroma.com/ -d 3 | cdp chunk -s 512| cdp tx emoji-clean -m | cdp embed --ef default
```
