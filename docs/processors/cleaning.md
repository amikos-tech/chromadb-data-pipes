# Cleaning Processors

## Clean Emoji

Usage:

```bash
cdp tx emoji-clean [-m] [-|<file>]
```

!!! note "Get Help"

    Get help for the command with the following flag:

    ```bash
    cdp tx emoji-clean --help
    ```

### Example

The following example cleans the ChromaDB docs home page from emojis, including metadata.

```bash
cdp imp url https://docs.trychroma.com/ -d 1 | cdp tx emoji-clean -m
```
