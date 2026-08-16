# Contributing to minigpt_llm

## 🛠️ Development Setup

1. **Fork & clone**
   ```bash
   git clone git@github.com:Ankur02Sarkar/minigpt.git
   cd minigpt
   ```

2. **Create a branch**
   ```bash
   git checkout -b phase/<N>-<slug> main
   # e.g., git checkout -b phase/7-dockerfile main
   ```

3. **Set up the environment**
   ```bash
   # Copy env example
   cp .env.example .env

   # Install uv + deps
   python3 -m venv .venv && source .venv/bin/activate
   uv pip install -r requirements.txt -r requirements-dev.txt

   # Install pre-commit hooks
   pre-commit install

   # Verify
   .venv/bin/python -m pytest -x --tb=short 2>&1 | tail -5
   ```

4. **Make your change**
   - Follow the [Conventional Commits] spec: `feat(phase7): add dockerfile`
   - Keep `SEED=42` for reproducibility
   - Run `ruff format` + `ruff check --fix` before committing
   - Run `mypy --strict model/ training/ serving/` type-check

5. **Test your change**
   - Add unit tests in `tests/` corresponding to your module
   - Ensure all existing tests pass: `.venv/bin/python -m pytest -x`
   - For serving changes, verify with both `openai-python` SDK and Ollama CLI

6. **Commit & PR**
   ```bash
   git add .
   git commit -m "feat(phase7): add dockerfile (closes #X)"
   git push origin phase/7-dockerfile
   ```

7. **Open a PR**
   - Fill the [PR template]{.filepath}`.github/PULL_REQUEST_TEMPLATE.md`
   - Ensure CI is green (lint + typecheck + test)
   - Tag a maintainer for review

## 📝 PR Checklist (per AGENTS.md §5 Definition of Done)
- [x] Code merged to `main` via PR with green CI
- [x] Tests covering the new behavior added
- [x] `README.md` updated if any §0.1 trigger conditions apply
- [x] `AGENTS.md` flipped `[x]` → `[ ]` or `[~]` as appropriate
- [x] `ARCHITECTURE.md` updated if architecture itself changed
- [x] `logs/<run>/` artifacts committed or linked from PR

## 🐞 Issue Templates

### Bug Report
Please include:
- Minigpt version / commit SHA
- OS & Python version
- Steps to reproduce
- Expected vs. actual behavior
- Relevant error logs / `STATUS.json` snapshot

### Feature Request
Please include:
- Use case / motivation
- Desired behavior
- Any relevant references or examples

## 📄 Pull Request Template
See `.github/PULL_REQUEST_TEMPLATE.md` for the structured template.

## 📜 Code of Conduct
This project adheres to the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code-of-conduct.html). Please read the full text in `.github/CODE_OF_CONDUCT.md` if present.

## 👥 Code Owners
```text
# Codeowners file content
/minigpt_llm/ @minigpt-maintainer
/training/ @minigpt-maintainer
/serving/ @minigpt-maintainer
```
