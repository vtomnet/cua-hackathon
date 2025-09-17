1. Set env vars: OPENAI_API_KEY (required), ANTHROPIC_API_KEY (optional), GROQ_API_KEY (optional)
2. `uv sync`
3. `source .venv/bin/activate && python3 -m computer_server` (note: must wait until it confirms it is listening)
4. `uv run main.py 2>&1 | tee out.log` (note: must wait until it confirms it is listening)
5. Start vad-app; see ../vad-app/README.md
