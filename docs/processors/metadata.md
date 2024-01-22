# Metadata

The library offers a way to easily manipulate metadata values by adding or updating existing metadata keys as well as
removing metadata keys from the metadata dictionary.

## Usage

```bash
cdp meta [-m key=value] [-k key_to_remove] [-o]
```

To add or update metadata key use `-m` flag with a `key=value` pair. The key is always assumed to be a string. The
value is processed as follows - boolean value (true/false), float value, integer value. If the value
cannot be parsed as any of the above types, it is assumed to be a string. If the value is with escaped
single-quotes `'` (e.g. `\'value\'`), the escaped single-quotes are removed and the value treated as a string.

To remove metadata keys use `-k` flag. If the key does not exist in the metadata dictionary, it is ignored.

Examples:

Add metadata keys:

```bash
cat sample-data/metadata/metadata.jsonl | head -1 | cdp meta -m bool_value=false -m int_value=1 -m float_value=1.1 -m escaped_value=\'true\' | jq .metadata
{
  "title": "Animalia (book)",
  "bool_value": false,
  "int_value": 1,
  "float_value": 1.1,
  "escaped_value": "true"
}
```

Overwrite metadata keys:

```bash
cat sample-data/metadata/metadata.jsonl | head -1 | cdp meta -m title="New Title" -o | jq .metadata
{
  "title": "New Title"
}
```

Remove metadata keys:

```bash
cat sample-data/metadata/metadata.jsonl | head -1 |  cdp meta -m bool_value=false -m int_value=1 | cdp meta -k int_value | jq .metadata
{
  "title": "Animalia (book)",
  "bool_value": false
}
```

!!! note "Help"

    Run `cdp chunk --help` for more information.
