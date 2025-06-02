set -euo pipefail

. scripts/variables.bash

DOCKER_BUILDKIT=1 docker build "$@" -t "$DOCKER_DEV_IMAGE":latest -f Dockerfile-dev .
