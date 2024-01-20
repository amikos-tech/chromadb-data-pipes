# Concepts

## EmbeddableResource

ChromaDB Data Pipes operates over a structure called EmbeddableResource. The structure is a closely related to ChromaDB
Documents. All core components of the library will either, produce, consume, or transform EmbeddableResources.

Each embeddable resource has the following properties:

- `id` - unique identifier of the resource
- `metadata` - metadata of the resource
- `embedding` - embedding of the resource

For text resource we use `EmbeddableTextResource` which adds the following properties:

- `text_chunk` - text of the resource

!!! note "Evolution"

    We plan to evolve the EmbeddableResource structure to support more types of resources, such as images, audio, video,

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

!!! note "WIP"

    This is a work in progress. Stay tuned for more updates.
