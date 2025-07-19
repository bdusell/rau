set -euo pipefail
exec bash --init-file <(echo '. ~/.bashrc && eval $(poetry env activate)')
