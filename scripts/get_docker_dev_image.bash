set -euo pipefail

. scripts/variables.bash

usage() {
  echo "Usage: $0 [options]

Ensure that the Docker image has been created.
  
Options:
  --build   Build the Docker image from scratch.
"
}

mode=none
while [[ $# -gt 0 ]]; do
  case $1 in
    --build) mode=build ;;
    *) usage >&2; exit 1 ;;
  esac
  shift
done

case $mode in
  none) ;;
  build) bash scripts/build_docker_dev_image.bash ;;
esac
