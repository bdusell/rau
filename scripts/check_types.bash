set -euo pipefail
if [[ $# -gt 0 ]]; then
  args=("$@")
else
  BASE_DIR=$(cd "$(dirname "$BASH_SOURCE")"/.. && pwd)
  args=("$BASE_DIR"/{src,tests})
fi
exec mypy "${args[@]}"
