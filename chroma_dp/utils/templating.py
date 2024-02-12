import time
from datetime import datetime
from typing import Any

from jinja2 import Environment


def date(date_format: str = "epoch") -> str:
    """Custom filter to format the date based on the input format."""
    if date_format == "epoch":
        return str(time.time())
    else:
        return datetime.now().strftime(date_format)


def now(*args: Any, **kwargs: Any) -> str:
    """Custom filter to return the current time."""
    return str(datetime.now())


def get_jinja_env() -> Environment:
    _env = Environment()
    _env.filters["date"] = date
    _env.filters["now"] = now
    return _env
