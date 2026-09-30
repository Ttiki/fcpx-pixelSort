#!/bin/bash
# Runs production Metal motion estimation/compositing against an independent CPU reference.
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -ne 1 ]]; then
  echo 'Usage: bash Scripts/test-datamosh.sh /path/to/default.metallib' >&2
  exit 2
fi
check_binary="$(mktemp -d)/datamosh-checks"
xcrun swiftc -O \
  -import-objc-header "$project_root/1V2T_Glitch/Plugin/TileableRemoteBrightnessShaderTypes.h" \
  "$project_root/1V2T_Glitch/Plugin/MetalDeviceCache.swift" \
  "$project_root/Tests/DatamoshGPUChecks.swift" -o "$check_binary"
"$check_binary" "$1"
# Exercise the production rational-time helper separately from the GPU and host APIs.
xcrun swiftc \
  "$project_root/1V2T_Glitch/Plugin/DatamoshTiming.swift" \
  "$project_root/Tests/DatamoshTimingChecks.swift" -o "${check_binary}-timing"
"${check_binary}-timing"
