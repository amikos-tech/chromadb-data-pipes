# Authentication

When importing or exporting data from remote Chroma instance, you may need to authenticate your API requests. CDP
supports auth via URL or env vars.

### URL


**Basic Auth:**

```bash
cdp import/export http://admin:admin@localhost:8000/my_collection
```

**Token Auth:**

```bash
cdp import/export http://__auth_token__:chr0ma-t0k3n@localhost:8000/my_collection
```

**X-Chroma Token Auth:**

```bash
cdp import/export http://__x_chroma_token__:chr0ma-t0k3n@localhost:8000/my_collection
```

### Environment Variables

**Basic Auth:**

```bash
export CHROMA_BASIC_AUTH=admin:admin
cdp import/export http://localhost:8000/my_collection
```

**Token Auth:**

```bash
export CHROMA_TOKEN_AUTH=chr0ma-t0k3n
cdp import/export http://localhost:8000/my_collection
```

**X-Chroma Token Auth:**

```bash
export CHROMA_XTOKEN_AUTH=chr0ma-t0k3n
cdp import/export http://localhost:8000/my_collection
```
