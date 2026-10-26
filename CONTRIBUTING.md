# Contributing

This project was developed by a single person with the help of modern LLM/AI coding tools. It's not under continuous active development, so **any contribution is particularly welcome** — whether it's a bug fix, documentation improvement, new feature, or just a question.

## How to Help

- **Bug reports**: Open an issue describing the problem
- **Code contributions**: Fork, make changes, submit a PR
- **Questions**: Feel free to open an issue

There are no strict style requirements. Just try to follow existing patterns where reasonable.

## Running the Tests

```bash
pip install -r requirements.txt pytest
python -m pytest
```

Tests that need Gurobi are skipped when it is not installed, so please make sure the suite passes without Gurobi too: the `heuristic` pipeline and the evaluator must keep working without it. See [docs/developer_notes.md](docs/developer_notes.md) for implementation details.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
