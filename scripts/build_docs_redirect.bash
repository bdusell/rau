set -euo pipefail
latest_version=$1
sed "
    s|{{NEW_URL}}|https://bdusell.github.io/rau/$latest_version/index.html|g
" docs/_redirect.html > docs/dist/index.html
