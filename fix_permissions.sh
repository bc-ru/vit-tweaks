#!/usr/bin/env bash
# Repair only generated project files; new jobs should use scripts/run.sh instead.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "$repo_root" == / || ! -d "$repo_root/src/aerial_svd" ]]; then
  echo "Refusing to repair outside this repository." >&2
  exit 1
fi
printf 'Repairing artifacts/, .scratch/, .cache/ for %s:%s\n' "$(id -u)" "$(id -g)"
docker run --rm --pull never --network none --read-only --user 0:0 \
  --cap-drop ALL --cap-add CHOWN --cap-add FOWNER --cap-add DAC_OVERRIDE \
  --security-opt no-new-privileges \
  --volume "$repo_root:/workspace" --workdir /workspace --interactive \
  aerial-svd-jax:cpu python - "$(id -u)" "$(id -g)" <<'PY'
import os
from pathlib import Path
import stat
import sys

uid, gid = map(int, sys.argv[1:])
changed = 0
for name in ("artifacts", ".scratch", ".cache"):
    root = Path("/workspace") / name
    if root.is_symlink():
        raise RuntimeError(f"Refusing symlink repair root: {root}")
    if not root.exists():
        continue
    for directory, dirs, files, descriptor in os.fwalk(root, follow_symlinks=False):
        info = os.fstat(descriptor)
        mode = stat.S_IMODE(info.st_mode) | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
        if (info.st_uid, info.st_gid) != (uid, gid) or mode != stat.S_IMODE(info.st_mode):
            os.fchown(descriptor, uid, gid)
            os.fchmod(descriptor, mode)
            changed += 1
        for entry in files + dirs:
            info = os.stat(entry, dir_fd=descriptor, follow_symlinks=False)
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                continue
            mode = stat.S_IMODE(info.st_mode) | stat.S_IRUSR | stat.S_IWUSR
            if stat.S_ISDIR(info.st_mode) or mode & 0o111:
                mode |= stat.S_IXUSR
            if (info.st_uid, info.st_gid) != (uid, gid) or mode != stat.S_IMODE(info.st_mode):
                child = os.open(entry, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
                try:
                    actual = os.fstat(child)
                    if not (stat.S_ISREG(actual.st_mode) or stat.S_ISDIR(actual.st_mode)):
                        raise RuntimeError(f"Unexpected file type during repair: {entry}")
                    os.fchown(child, uid, gid)
                    os.fchmod(child, mode)
                    changed += 1
                finally:
                    os.close(child)
print(f"Done: repaired {changed} entries; symlinks were not followed.")
PY
