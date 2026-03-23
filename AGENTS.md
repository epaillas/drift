# Repository Guidelines

## Project Structure & Module Organization

Core package code lives in [`drift/`](/Users/epaillas/code/drift/drift). Key modules include theoretical models in `models.py`, `eft_models.py`, and `galaxy_models.py`, loop integrals in `one_loop.py`, analytic emulators in `emulator.py` and `galaxy_emulator.py`, and supporting utilities such as `cosmology.py`, `kernels.py`, and `multipoles.py`. Tests live in [`tests/`](/Users/epaillas/code/drift/tests) and generally mirror module names, for example `tests/test_eft_models.py`. Project documentation is in [`docs/`](/Users/epaillas/code/drift/docs), and runnable analysis or plotting entry points are in [`scripts/`](/Users/epaillas/code/drift/scripts). Example runtime configuration is in [`configs/example.yaml`](/Users/epaillas/code/drift/configs/example.yaml).

## Build, Test, and Development Commands

Install the package in editable mode with:

```bash
pip install -e .
```

Run the full test suite with:

```bash
pytest
```

Run a focused subset while iterating:

```bash
pytest tests/test_eft_models.py tests/test_emulator.py
```

Build a distributable package with:

```bash
python -m build
```

Use scripts for manual validation or plotting, for example `python scripts/inference_pgg.py` or `python scripts/test_emulator_accuracy.py`.

## Coding Style & Naming Conventions

Use 4-space indentation and standard Python naming: `snake_case` for functions and variables, `CamelCase` for classes, and `UPPER_CASE` for module constants. Keep numerical code vectorized with NumPy where practical, and document non-obvious formulas with short docstrings or inline comments. Follow the existing pattern of small, single-purpose modules and tests that mirror public behavior.

## Testing Guidelines

This project uses `pytest`. Add tests for every behavioral change, especially for emulator/direct-model consistency, EFT parameter activation, and loop-term edge cases. Name tests `test_<behavior>()` and keep fixtures local to each test module unless widely reused. Prefer targeted regression tests over broad integration-only coverage.

## Commit & Pull Request Guidelines

Recent history uses short, imperative commit subjects such as `Add one-loop galaxy bias...` or `Update model source files...`. Follow that style: start with a verb and summarize the user-visible change. PRs should include a concise description, note affected modules, mention any numerical or API changes, and list the exact test command(s) run. Include plots only when a plotting script or visual output changed.
