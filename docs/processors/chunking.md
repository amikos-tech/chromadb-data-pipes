# Chunking

To chunk documents we have a chunk processor - `cdp tx chunk` that can be used as follows:

```bash
cdp imp pdf sample-data/papers/ | grep "2401.02412.pdf" | head -1 | cdp tx chunk -s 500 -a
```

The above will chunk the document into 500 character chunks and print the chunks to stdout. We will also add (`-a`
option) the offset position of each chunk within the document as metadata `start_index`.

!!! note "Help"

    Run `cdp tx chunk --help` for more information.
