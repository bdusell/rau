set -e
set -u

. scripts/variables.bash

DOCKER_BUILDKIT=1 docker build "$@" -t "$DOCKER_DEV_IMAGE":latest -f Dockerfile-dev .
