exec bash scripts/poetry_run.bash bash -c '
  set -euo pipefail
  pytest tests
  bash tests/test_tutorial.bash
'
