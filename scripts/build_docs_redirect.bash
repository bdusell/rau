set -euo pipefail
sed "
    s|{{NEW_URL}}|https://bdusell.github.io/rau/main/index.html|g
" docs/_redirect.html > docs/dist/index.html
