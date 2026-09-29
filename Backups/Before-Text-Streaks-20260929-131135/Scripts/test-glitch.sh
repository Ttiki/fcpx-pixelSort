#!/bin/bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash Scripts/test-glitch.sh /path/to/default.metallib [preview.png]' >&2
  exit 2
fi
check_binary="$(mktemp -d)/glitch-gpu-checks"
xcrun swiftc -O \
  -import-objc-header "$project_root/PixelSort/Plugin/TileableRemoteBrightnessShaderTypes.h" \
  "$project_root/PixelSort/Plugin/MetalDeviceCache.swift" \
  "$project_root/Tests/GlitchGPUChecks.swift" \
  -o "$check_binary"
"$check_binary" "$@"
