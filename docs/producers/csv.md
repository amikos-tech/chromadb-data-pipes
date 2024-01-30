# CSV Files

## Default CSV File Generator

The default CSV file generator reads a single CSV file provided as argument to `cdp imp csv` command.

Practical example:

```bash
 cdp imp csv sample-data/csv/employees_with_resumes.csv  \
  --meta-features Name \
  --meta-features Department \
  --doc-feature Resume \
  | tail -1 \
  | cdp embed --ef default
```

The above command will read the CSV file `sample-data/csv/employees_with_resumes.csv` and will use the columns `Name`
and `Department` as metadata features and the column `Resume` as document feature. The output of the command is piped
to `cdp embed` which will embed the document feature using the default embedding function.

### Options

- `--meta-features` - a list of columns to be used as metadata features (specified multiple times, once per column)
- `--doc-feature` - a column to be used as document feature. If unspecified, the entire row is used as document feature.
- `--batch-size` - the number of rows to sbe read at once. Defaults to 100.
- `--delimiter` - the delimiter used in the CSV file. Defaults to `,`.
- `--quotechar` - the quote character used in the CSV file. Defaults to `"`.

!!! note "Help"

    Run `cdp imp csv --help` for more information.
