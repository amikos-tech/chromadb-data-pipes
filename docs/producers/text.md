# Text Files

## Default Text File Generator

Reading a dir with text files to stdout:

```bash
cdp imp txt sample-data/text/ | head -1 | cdp chunk -s 100 | tail -1 | cdp embed --ef default
```

!!! note "Help"

    Run `cdp imp txt --help` for more information.
