FROM python:3.11-slim-bookworm as final

COPY ./ /app

WORKDIR /app

# install poetry
RUN pip install poetry && \
    poetry update --no-interaction --no-ansi --with=docs --with=embeddings --with=web

ENTRYPOINT ["poetry", "run"]
CMD ["cdp","--help"]
