set -euo pipefail

. scripts/variables.bash

bash scripts/get_docker_dev_image.bash "$@"
singularity build "$SINGULARITY_IMAGE".sif docker-daemon://"$DOCKER_DEV_IMAGE":latest
