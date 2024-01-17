# CDP Concepts

## Producer

Generates a stream of data to a file or stdout.

The source of the data is implementation dependent, HF datasets, ChromaDB, file etc.

## Consumer

Consumes a stream of data from a file or stdin.

## Processor

Consumes a stream of data from a file or stdin and processes it by some criteria. Produces a stream of data to a file or
stdout.

## Pipeline

Reusable set of producer, consumer, filter, and transformer.

Properties:
- Variables
