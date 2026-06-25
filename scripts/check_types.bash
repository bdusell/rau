set -euo pipefail
if [[ $# -gt 0 ]]; then
  args=("$@")
else
  base_dir=$(dirname "$BASH_SOURCE")/..
  args=("$base_dir"/src "$base_dir"/tests)
fi
mypy "${args[@]}"
