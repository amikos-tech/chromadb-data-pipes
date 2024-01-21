import sys
from contextlib import contextmanager
from typing import Generator, Optional, TextIO, Union, IO, Any


@contextmanager
def smart_open(
    filename: Optional[str] = None,
    stdin: TextIO = sys.stdin,
    mode: str = "r",
) -> Generator[Union[IO[Any], TextIO], None, None]:
    fh: Union[IO[Any], TextIO] = stdin
    if filename:
        fh = open(filename, mode)
    try:
        yield fh
    finally:
        if filename and isinstance(fh, IO):
            fh.close()
