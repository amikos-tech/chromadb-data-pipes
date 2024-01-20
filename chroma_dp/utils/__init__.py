import sys
from contextlib import contextmanager
from typing import Generator, Optional, TextIO, Union, IO, Any


@contextmanager
def smart_open(
    filename: Optional[str] = None,
    stdin: Optional[TextIO] = sys.stdin,
    mode: str = "r",
) -> Generator[Union[Optional[IO[Any]], Optional[TextIO]], None, None]:
    fh: Union[Optional[IO[Any]], Optional[TextIO]] = stdin
    if filename:
        fh = open(filename, mode)
    try:
        yield fh
    finally:
        if filename and isinstance(fh, IO):
            fh.close()
