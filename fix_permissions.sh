#!/usr/bin/env bash
set -euo pipefail

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

# Safety guard: do not allow running from dangerous broad directories.
case "$(pwd)" in
  /|/home|/root|/usr|/etc|/var|/opt)
    echo "Refusing to run from unsafe directory: $(pwd)" >&2
    exit 1
    ;;
esac

echo "Fixing ownership under: $(pwd)"
echo "New owner: ${HOST_UID}:${HOST_GID}"

sudo chown -R "${HOST_UID}:${HOST_GID}" .
chmod -R u+rwX .

echo "Done."