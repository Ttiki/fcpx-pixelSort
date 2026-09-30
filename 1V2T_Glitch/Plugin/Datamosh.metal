// A temporal datamosh-style renderer: match blocks between adjacent frames, then reuse delayed
// pixels with exaggerated backward motion. This operates on decoded frames, not codec bitstreams.
#include <metal_stdlib>
#include "TileableRemoteBrightnessShaderTypes.h"
using namespace metal;

/// Reads an image-anchored sample with transparent borders and the frame's own tile/origin.
float4 moshRead(texture2d<float, access::read> image, int2 point, int4 rect, int top, int2 size) {
    int2 local = point - rect.xy;
    if (any(point < int2(0)) || any(point >= size) || any(local < int2(0)) || any(local >= rect.zw)) return float4(0);
    if (top != 0) local.y = rect.w - 1 - local.y;
    return image.read(uint2(local));
}

/// Keeps a stable set of corrupted blocks for a given seed, independent of frame render order.
uint moshHash(uint value) {
    value ^= value >> 16; value *= 0x7feb352du;
    value ^= value >> 15; value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

/// Measures sparse RGBA error for a proposed displacement using nine samples within the block.
/// Premultiplied color and alpha both contribute, allowing motion of transparent text to be detected.
float moshCost(texture2d<float, access::read> current, texture2d<float, access::read> previous,
               int2 origin, int2 displacement, constant DatamoshUniforms &u) {
    float cost = 0;
    for (int y = 1; y <= 3; ++y) for (int x = 1; x <= 3; ++x) {
        int2 point = min(origin + int2(x, y) * u.imageInfo.z / 4, u.imageInfo.xy - 1);
        float4 a = moshRead(current, point, u.currentRect, u.origins.x, u.imageInfo.xy);
        float4 b = moshRead(previous, point + displacement, u.previousRect, u.origins.y, u.imageInfo.xy);
        cost += dot(abs(a-b), float4(0.2126f, 0.7152f, 0.0722f, 0.25f));
    }
    return cost;
}

/// Finds the best backward displacement on a bounded 9x9 candidate grid for each output block.
/// Flat/equal-cost areas prefer zero or shorter motion rather than inventing a directional bias.
kernel void datamoshMotion(texture2d<float, access::read> current [[texture(0)]],
                            texture2d<float, access::read> previous [[texture(1)]],
                            device int2 *vectors [[buffer(0)]],
                            constant DatamoshUniforms &u [[buffer(1)]],
                            uint2 index [[thread_position_in_grid]]) {
    if (any(index >= uint2(u.grid.zw))) return;
    int2 best = int2(0);
    if (u.controls.x > 0 && u.controls.y > 0 && u.controls.z > 0) {
        int2 origin = (u.grid.xy + int2(index)) * u.imageInfo.z;
        float bestCost = moshCost(current, previous, origin, best, u);
        for (int y = -4; y <= 4; ++y) for (int x = -4; x <= 4; ++x) {
            int2 candidate = int2(x,y) * u.imageInfo.w;
            float cost = moshCost(current, previous, origin, candidate, u);
            if (cost < bestCost || (cost == bestCost && dot(float2(candidate),float2(candidate)) < dot(float2(best),float2(best)))) {
                bestCost = cost; best = candidate;
            }
        }
    }
    vectors[index.y * uint(u.grid.z) + index.x] = best;
}

/// Replaces a seeded proportion of blocks with displaced delayed-frame pixels and applies Mix.
/// With Motion zero it becomes temporal block freezing; Amount/Mix zero are exact bypasses.
kernel void datamoshComposite(texture2d<float, access::read> current [[texture(0)]],
                               texture2d<float, access::read> history [[texture(1)]],
                               texture2d<float, access::write> output [[texture(2)]],
                               const device int2 *vectors [[buffer(0)]],
                               constant DatamoshUniforms &u [[buffer(1)]],
                               uint2 index [[thread_position_in_grid]]) {
    if (any(index >= uint2(u.destinationRect.zw))) return;
    int2 point = int2(index) + u.destinationRect.xy;
    float4 original = moshRead(current, point, u.currentRect, u.origins.x, u.imageInfo.xy);
    float4 result = original;
    if (u.controls.x > 0 && u.controls.z > 0) {
        int2 block = point / u.imageInfo.z;
        uint key = moshHash(uint(block.x) ^ moshHash(uint(block.y)) ^ moshHash(u.random.x));
        float chance = float(key & 0x00ffffffu) / 16777216.0f;
        if (chance < u.controls.x) {
            int2 local = block - u.grid.xy;
            int2 vector = vectors[local.y * u.grid.z + local.x];
            int2 displaced = point + int2(round(float2(vector) * u.controls.y * 8.0f));
            result = moshRead(history, displaced, u.historyRect, u.origins.z, u.imageInfo.xy);
            if (u.controls.z < 1) result = mix(original, result, u.controls.z);
        }
    }
    uint2 target = index;
    if (u.origins.w != 0) target.y = uint(u.destinationRect.w) - 1 - target.y;
    output.write(result, target);
}
