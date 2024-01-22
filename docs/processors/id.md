# ID Generation

CDP provides a number of ID generation strategies that can be used to generate or regenerate IDs for a given dataset.

The following strategies are available:

- [UUID](#uuid) - IDS are generated using the UUID
- [ULID](#ulid) - IDS are generated using the ULID
- [Document Hash](#document-hash) - IDS are generated using the document hash (SHA256)
- [Random Hash](#random-hash) - IDS are generated using a random hash (SHA256)
- [Expression](#expression) - IDS are generated using a Jinja2 expression

## Usage

!!! note "Help"

    Run `cdp id --help` for more information.

### UUID

This strategy generates a unique ID based on UUIDv4.

```bash
cat sample-data/metadata/metadata.jsonl | head -1 | cdp id --uuid | jq '.id'
```

Returns:

```json
"5bf1d91b-817c-47a2-a6cb-94894c1b42c3"
```

### ULID

This strategy generates a unique ID based on [ULID](https://pypi.org/project/py-ulid/).

```bash
cat sample-data/metadata/metadata.jsonl | head -1 | cdp id --ulid | jq '.id'
```

Returns:

```json
"01HMR7V5MMVD3Q1PA5PPSF0FA1"
```

### Document Hash

This strategy generates unique IDs based on the document hash (SHA256).

```bash
cat sample-data/metadata/metadata.jsonl | head -1 | cdp id --doc-hash | jq '.id'
```

Returns:

```json
"3143643f8520f32f7b04fff2cd524acbe32ef989b2bd6cc89d687743a909bfa6"
```

### Random Hash

This strategy generates unique IDs based on a random hash (SHA256).

```bash
cat sample-data/metadata/metadata.jsonl | head -1 | cdp id --random-hash | jq '.id'
```

Returns:

```json
"f4d7bc16f4ddbd080b08c4836efa93ed51e85ea1289df6c5851e261893e6ad52"
```

### Expression

Generates ID based on provided [Jinja2](https://pypi.org/project/Jinja2/) expression. The following variables are available for use in the expression:

- `metadata` - the metadata for the document
- `text_chunks` - the text chunks for the document
- `id` - existing ID for the document
- `embedding` - the embedding for the document
- `uuid` - function that generates a UUID (example usage `{{uuid()}}`)
- `ulid` - function that generates a ULID (example usage `{{ulid()}}`)

```bash
cat sample-data/metadata/metadata.jsonl | head -1 | cdp id --expr '{{ulid()}}-{{metadata.title}}' | jq '.id'
```

Returns:

```json
"01HMR8308TMG80XYV9CHNAF0A3-Animalia (book)"
```
