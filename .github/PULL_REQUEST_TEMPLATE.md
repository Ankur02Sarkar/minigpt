---
description: Pull Request Template for minigpt_llm
---
## 📥 Pull Request Template

### Summary
**What does this PR do?** _(be specific — link to issue if applicable)_\
**Which phase does this affect?** _(e.g., Phase 7, Phase 5.3)_\
**Breaking changes?** _(yes/no + description if yes)_\

### Checklist (per AGENTS.md §5)
- [ ] Code merged to `main` via PR with green CI
- [ ] Tests covering new behavior added/updated
- [ ] `README.md` updated if §0.1 trigger conditions apply
- [ ] `AGENTS.md` phase markers flipped as appropriate
- [ ] `ARCHITECTURE.md` updated if architecture changed
- [ ] `logs/<run>/` artifacts committed or linked

### Type of Change
- [ ] Bug fix (non-breaking correction)
- [ ] New feature (non-breaking addition)
- [ ] Breaking change / deprecation
- [ ] Documentation update
- [ ] CI / tooling / infrastructure

### Testing
- [ ] `uv run ruff format` passes
- [ ] `uv run ruff check --fix` passes
- [ ] `uv run mypy --strict model/ training/ serving/` passes
- [ ] New/updated tests pass: `uv run pytest -x`
- [ ] Manual verification (describe what you tested)

### Screenshots / Logs
_(attach if applicable — e.g., STATUS.json, loss curves, error outputs)_
