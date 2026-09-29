// GPU image operations for Signal Tear and RGB Split. Each thread computes one output pixel
// from image-anchored coordinates, making the result independent of how the host tiles a frame.
// metal_stdlib provides vectors and GPU texture access; the shared header matches Swift's ABI.

#include <metal_stdlib>
#include "TileableRemoteBrightnessShaderTypes.h"
using namespace metal;

/// Mixes integer seed/time/band keys into reproducible pseudo-random bits using unsigned wraparound.
uint glitchHash(uint value) {
    value ^= value >> 16; value *= 0x7feb352du;
    value ^= value >> 15; value *= 0x846ca68bu;
    return value ^ (value >> 16);
}
/// Maps 24 random bits to [0,1), a range exactly representable with float mantissa precision.
float glitchRandom(uint value) { return float(glitchHash(value) & 0x00ffffffu) / 16777216.0f; }

/// Samples a source pixel with transparent borders and converts image coordinates to texture rows.
/// Explicit bounds checks prevent wraparound or out-of-tile reads from creating false edge pixels.
float4 glitchRead(texture2d<float, access::read> source, constant GlitchUniforms &u, int2 point) {
    int2 local = point - u.sourceRect.xy;
    if (any(point < int2(0)) || any(point >= u.imageInfo.xy) ||
        any(local < int2(0)) || any(local >= u.sourceRect.zw)) return float4(0);
    if (u.imageInfo.z != 0) local.y = u.sourceRect.w - 1 - local.y;
    return source.read(uint2(local));
}

/// Displaces bands or samples separate RGB positions, then applies the shared Mix control.
/// The output alpha tracks moved coverage so the effects also work on transparent text.
kernel void glitchEffect(texture2d<float, access::read> source [[texture(0)]],
                         texture2d<float, access::write> destination [[texture(1)]],
                         constant GlitchUniforms &u [[buffer(0)]],
                         uint2 position [[thread_position_in_grid]]) {
    if (any(position >= uint2(u.destinationRect.zw))) return;
    int2 point = int2(position) + u.destinationRect.xy;
    float4 original = glitchRead(source, u, point);
    float4 result = original;
    if (u.controls.x > 0 && u.controls.z > 0) {
        // Reconstruct the random key from immutable uniforms rather than render-call order.
        uint base = u.configuration.y ^ glitchHash(u.configuration.z + 0x9e3779b9u);
        if (u.configuration.x == 0) {
            int height = max(1, int(round(float(u.imageInfo.y) * (0.005f + 0.12f * u.controls.y))));
            // Global image Y makes adjacent output tiles choose identical bands.
            uint band = uint(point.y / height);
            uint key = base ^ glitchHash(band + 17u);
            if (glitchRandom(key) < u.controls.w) {
                float displacement = (2.0f * glitchRandom(key ^ 0xa511e9b3u) - 1.0f) * u.controls.x * float(u.imageInfo.x) * 0.25f;
                int shift = int(round(displacement));
                result = glitchRead(source, u, point - int2(shift, 0));
            }
        } else {
            float pulse = u.configuration.w != 0 ? 0.25f + 0.75f * glitchRandom(base) : 1.0f;
            int2 shift = int2(round(u.offset.xy * pulse));
            float4 red = glitchRead(source, u, point - shift);
            float4 blue = glitchRead(source, u, point + shift);
            // Channels are already premultiplied: keep their coverage in the composite alpha.
            // This makes colored fringes visible outside white text on transparent backgrounds.
            result = float4(red.r, original.g, blue.b, max(red.a, max(original.a, blue.a)));
        }
        if (u.controls.z < 1) result = mix(original, result, u.controls.z);
    }
    uint2 output = position;
    if (u.imageInfo.w != 0) output.y = uint(u.destinationRect.w) - 1 - output.y;
    destination.write(result, output);
}
