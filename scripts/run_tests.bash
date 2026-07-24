set -euo pipefail
test_file=${1-$(dirname "$BASH_SOURCE")/../tests}
shift 1 || true
case $test_file in
  *.py) pytest "$test_file" "$@" ;;
  *.bash) bash "$test_file" "$@" ;;
  *)
    pytest "$test_file" "$@"
    find "$test_file" -name '*.bash' -print -exec bash '{}' ';'
    ;;
esac
