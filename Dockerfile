FROM python:3.11-slim-bookworm as final

COPY ./ /app

WORKDIR /app

# install poetry
RUN pip install poetry && \
    poetry install --no-dev --no-interaction --no-ansi

ENTRYPOINT ["poetry", "run"]
