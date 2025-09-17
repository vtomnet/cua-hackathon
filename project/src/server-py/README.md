## Init

1. Set env vars: OPENAI_API_KEY (required), ANTHROPIC_API_KEY (optional), GROQ_API_KEY (optional)
2. `uv sync`

## Run

```bash
uv run python -m computer_server
uv run main.py 2>&1 | tee out.log
```
