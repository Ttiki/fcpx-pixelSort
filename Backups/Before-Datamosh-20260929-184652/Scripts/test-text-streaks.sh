#!/bin/bash
# Compile independent CPU/GPU comparisons against the production encoder and shared ABI.
# A built Metal library is supplied explicitly to avoid testing stale or unrelated shaders.
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash Scripts/test-text-streaks.sh /path/to/default.metallib [preview.png]' >&2
  exit 2
fi
# Temporary binaries stay outside the repository, leaving Xcode sources unchanged.
check_binary="$(mktemp -d)/text-streak-checks"
xcrun swiftc -O \
  -import-objc-header "$project_root/PixelSort/Plugin/TileableRemoteBrightnessShaderTypes.h" \
  "$project_root/PixelSort/Plugin/MetalDeviceCache.swift" \
  "$project_root/Tests/TextStreakChecks.swift" \
  -o "$check_binary"
"$check_binary" "$@"
