set -euo pipefail
sed "
    s|{{BASE_URL}}|https://bdusell.github.io/rau|g;
    s|{{NEW_PATH}}|/main/index.html|g
" docs/_redirect.html > docs/dist/index.html
