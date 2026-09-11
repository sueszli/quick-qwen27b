.PHONY: fmt
fmt:
	uvx ruff check --fix --line-length 5000 --extend-select I qwen.py example.py tests
	uvx ruff format --line-length 5000 qwen.py example.py tests

.PHONY: lint
lint:
	uv run --with pyright pyright qwen.py

.PHONY: tests
tests:
	uv run pytest tests/

.PHONY: example
example:
	uv run example.py

.PHONY: precommit
precommit:
	uv sync
	$(MAKE) fmt
	$(MAKE) lint
	$(MAKE) tests
