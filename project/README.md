```bash
uv sync
uv run main.py
curl -X POST http://localhost:8001/process -H "Content-Type: application/json" -d '{"text": "Do a rickroll"}'
```