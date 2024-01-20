# CLI Naming Conventions

Date: 2024-01-20
Tags: cli, naming, conventions
Authors: [@tazarov](https://github.com/tazarov)

## Context and Problem Statement

We want to be mindful of the naming conventions we use for the CLI. We want to be consistent and clear about the
commands we use and the order of the arguments.

## Considered Options

- Extensive use of subcommands
- Ecosystem centric naming

## Decision Outcome

Chosen option: "Ecosystem centric naming", because it will make the CLI more aligned with the ecosystem thus reducing
the cognitive load on users.

### The use of `import` and `export` for Chroma

This is a toolchain in the chroma ecosystem and we want to be very explicit about the use of import and export. They
must convey the core idea behind CDP, getting data in and out of Chroma.

### The use of `embed`

A core concept around vector databases is embedding. We want to make the embedding a first class citizen of CDP this is
why we choose to have embed as a top level command.

### The use of `chunk`

Another core concept around vector databases and embedding models is context window. Each embedding model has a specific
context window and sometimes the embedding models, make a trade-off to truncate data for better usability rather than
warning or preventing the user from using the model with an oversized context window. To that end chunking (of mostly
text data) has become a core concept in the ecosystem. We want to convey that by elevating the chunk command to a top
level command.

### The use of `ds-get` and `ds-put`

Working with datasets is another core concept in the econsystem. While we currently only plan to support HuggingFace
datasets, we want to also put the datasets front and center in CDP CLI. This is why we choose to have `ds-get`
and `ds-put` as top level commands.


## Pros and Cons of the Options

### Extensive use of subcommands

- Good, because it allows for a more natural grouping of commands
- Good, because it will solve a top-level command sprawl
- Bad, because it will put a burden on the user to remember and navigate the subcommands


### Ecosystem centric naming

- Good, because it will make the CLI more consistent with the ecosystem
- Good, because it reduces the cognitive load on users by aligning with expectations set by the ecosystem
- Bad, because it will increase the number of top-level commands