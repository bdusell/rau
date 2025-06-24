set -euo pipefail
latest_version=$( \
    sphinx-multiversion --dump-metadata docs docs/dist | \
    python scripts/get_latest_version.py \
)
sphinx-multiversion -D smv_latest_version="$latest_version" docs docs/dist
bash scripts/build_docs_redirect.bash "$latest_version"