set -euo pipefail
BASE_DIR=$(cd "$(dirname "$BASH_SOURCE")"/.. && pwd)
PYTHONPATH=$BASE_DIR/src exec poetry shell
