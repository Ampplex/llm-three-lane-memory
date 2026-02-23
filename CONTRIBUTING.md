# Contributing to threelane-memory

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/ankeshkumar/threelane-memory.git
cd threelane-memory

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode with dev extras
pip install -e ".[dev]"

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your Neo4j and OpenAI credentials
```

## Running Tests

```bash
# Unit tests (no API keys needed)
pytest tests/ -m "not integration"

# All tests including integration (requires .env)
pytest tests/
```

## Code Quality

```bash
# Lint
ruff check .

# Format
ruff format .

# Type check
mypy threelane_memory/
```

## Project Structure

```
threelane_memory/       # Core package
├── __init__.py         # Public API (store, query, close)
├── cli.py              # CLI entry point
├── config.py           # Configuration from .env
├── schemas.py          # TypedDict data contracts
├── database.py         # Neo4j driver wrapper
├── embeddings.py       # OpenAI embeddings + cosine similarity
├── llm_interface.py    # LLM chat wrapper
├── operator.py         # Semantic extraction
├── reconciler.py       # Graph writer + entity resolution
├── retriever.py        # 3-lane retrieval engine
├── entity_dedup.py     # Entity deduplication utility
├── backup.py           # Graph export to JSON
├── chat.py             # Interactive chat loop
└── utils.py            # Shared helpers
examples/               # Example scripts and optional UIs
tests/                  # Test suite
```

## Making Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest tests/ -m "not integration"`
5. Run linting: `ruff check .`
6. Commit with clear messages: `git commit -m "Add feature X"`
7. Push and open a Pull Request

## Guidelines

- Keep PRs focused — one feature or fix per PR
- Add tests for new functionality
- Update docstrings for public API changes
- Follow existing code style (enforced by ruff)
- Use type hints for function signatures

## Reporting Issues

- Use GitHub Issues
- Include Python version, OS, and Neo4j version
- Provide minimal reproduction steps
- Include error tracebacks when applicable

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
