.PHONY: sync test lint format notebook
sync:
	uv sync
test:
	uv run pytest
lint:
	uv run ruff check src tests experiments
format:
	uv run black src tests experiments
notebook:
	uv run jupyter lab