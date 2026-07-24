set -euo pipefail
BASE_DIR=$(cd "$(dirname "$BASH_SOURCE")"/.. && pwd)
PYTHONPATH=$BASE_DIR/src exec bash --init-file <(echo 'if [[ -e ~/.bashrc ]]; then . ~/.bashrc; fi && eval $(poetry env activate)')
