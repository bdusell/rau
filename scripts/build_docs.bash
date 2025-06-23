set -euo pipefail
sphinx-multiversion docs docs/dist
bash scripts/build_docs_redirect.bash