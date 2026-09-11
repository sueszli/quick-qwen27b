.PHONY: fmt
fmt:
	uvx ruff check --fix --line-length 5000 --extend-select I qwen.py example.py
	uvx ruff format --line-length 5000 qwen.py example.py

.PHONY: lint
lint:
	uv run --with pyright pyright qwen.py

.PHONY: example
example:
	uv run example.py

.PHONY: precommit
precommit:
	uv sync
	$(MAKE) fmt
	$(MAKE) lint
