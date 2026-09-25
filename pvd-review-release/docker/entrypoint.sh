#!/bin/sh
set -eu

command_name="${1:-smoke}"
if [ "$#" -gt 0 ]; then
    shift
fi

case "$command_name" in
    smoke)
        exec python -m path_opd.cli smoke \
            --work-dir /workspace/artifacts/toy-smoke \
            "$@"
        ;;
    test)
        cd /opt/path-opd
        exec python -m pytest -q "$@"
        ;;
    audit)
        exec python /opt/path-opd/scripts/audit_anonymity.py \
            /opt/path-opd \
            --no-git \
            "$@"
        ;;
    qualification)
        exec python /opt/path-opd/scripts/qualification.py \
            --root /opt/path-opd \
            --skip-runtime \
            "$@"
        ;;
    benchmark)
        exec python -m path_opd.cli benchmark "$@"
        ;;
    help|-h|--help)
        cat <<'EOF'
Usage: path-opd-entrypoint [smoke|test|audit|qualification|benchmark] [arguments]

  smoke  Run the deterministic CPU reproduction (default).
  test   Run the release test suite.
  audit  Scan the files copied into the image for anonymity leaks.
  qualification  Inspect release gates without benchmark dependencies.
  benchmark  Inspect contracts or preflight explicitly supplied assets.
EOF
        ;;
    *)
        echo "unknown container command: $command_name" >&2
        exit 64
        ;;
esac
