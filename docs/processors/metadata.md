# Metadata

The library offers a way to easily manipulate metadata values by adding or updating existing metadata keys as well as
removing metadata keys from the metadata dictionary.

## Usage

!!! warning "Deprecation"

        We are removing `-m` and `--meta` flags as they break from our end goal of consistent and non-ambiguous CLI flags.

```bash
cdp meta [-a key=value] [-k key_to_remove] [-o]
```

To add or update metadata key use `-a` flag with a `key=value` pair. The key is always assumed to be a string. The
value is processed as follows - boolean value (true/false), float value, integer value. If the value
cannot be parsed as any of the above types, it is assumed to be a string. If the value is with escaped
single-quotes `'` (e.g. `\'value\'`), the escaped single-quotes are removed and the value treated as a string.

To remove metadata keys use `-k` flag. If the key does not exist in the metadata dictionary, it is ignored.

!!! note "Help"

    Run `cdp meta --help` for more information.

### Add metadata keys

```bash
cat sample-data/metadata/metadata.jsonl | head -1 | cdp meta -a bool_value=false -a int_value=1 -a float_value=1.1 -a escaped_value=\'true\' | jq .metadata
```

Returns:

```json
{
  "title": "Animalia (book)",
  "bool_value": false,
  "int_value": 1,
  "float_value": 1.1,
  "escaped_value": "true"
}
```

### Overwrite metadata keys

```bash
cat sample-data/metadata/metadata.jsonl | head -1 | cdp meta -a title="New Title" -o | jq .metadata
```

Returns:

```json
{
  "title": "New Title"
}
```

### Remove metadata keys

```bash
cat sample-data/metadata/metadata.jsonl | head -1 |  cdp meta -a bool_value=false -a int_value=1 | cdp meta -k int_value | jq .metadata
```

Returns:

```json
{
  "title": "Animalia (book)",
  "bool_value": false
}
```

### Templating

It is also possible to pass template values to metadata values. The template values must be valid JINJA2 templates.

```bash
cat sample-data/metadata/metadata.jsonl | head -1 | cdp meta -a extracted_title="{{ metadata.title | upper}}" | jq .metadata
```

Returns:

```json
{
  "title": "Animalia (book)",
  "extracted_title": "ANIMALIA (BOOK)"
}
```

The following context vars and functions are available in the template:

- `metadata`: the metadata dictionary of the original doc
- `text_chunk`: the text chunk of the original doc
- `now`: the current datetime. Example usage `{{ now }}`
- `date`: Date in specified format, if not specified epoch time is returned. Example usage `{{ '%Y-%m-%d'| date }}`
- All the default [jinja2 filters.](https://jinja.palletsprojects.com/en/3.1.x/templates/#list-of-builtin-filters)
