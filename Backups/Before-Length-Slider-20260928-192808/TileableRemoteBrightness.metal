//
//  TileableRemoteBrightness.metal
//  PixelSort
//
//  Created by Clément Combier on 28/09/2026.
//

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

#include "TileableRemoteBrightnessShaderTypes.h"

typedef struct
{
    // The [[position]] attribute of this member indicates that this value is the clip space
    // position of the vertex when this structure is returned from the vertex function
    float4 clipSpacePosition [[position]];
    
    // Since this member does not have a special attribute, the rasterizer interpolates
    // its value with the values of the other triangle vertices and then passes
    // the interpolated value to the fragment shader for each fragment in the triangle
    float2 textureCoordinate;
    
} RasterizerData;

vertex RasterizerData
vertexShader(uint vertexID [[vertex_id]],
             constant Vertex2D *vertexArray [[buffer(BVI_Vertices)]],
             constant vector_uint2 *viewportSizePointer [[buffer(BVI_ViewportSize)]])
{
    RasterizerData out;
    
    // Index into our array of positions to get the current vertex
    //   Our positions are specified in pixel dimensions (i.e. a value of 100 is 100 pixels from
    //   the origin)
    float2 pixelSpacePosition = vertexArray[vertexID].position.xy;
    
    // Get the size of the drawable so that we can convert to normalized device coordinates,
    float2 viewportSize = float2(*viewportSizePointer);
    
    // The output position of every vertex shader is in clip space (also known as normalized device
    //   coordinate space, or NDC). A value of (-1.0, -1.0) in clip-space represents the
    //   lower-left corner of the viewport whereas (1.0, 1.0) represents the upper-right corner of
    //   the viewport.
    
    // In order to convert from positions in pixel space to positions in clip space we divide the
    //   pixel coordinates by half the size of the viewport.
    out.clipSpacePosition.xy = pixelSpacePosition / (viewportSize / 2.0);
    
    // Set the z component of our clip space position 0 (since we're only rendering in
    //   2-Dimensions for this sample)
    out.clipSpacePosition.z = 0.0;
    
    // Set the w component to 1.0 since we don't need a perspective divide, which is also not
    //   necessary when rendering in 2-Dimensions
    out.clipSpacePosition.w = 1.0;
    
    // Pass our input textureCoordinate straight to our output RasterizerData. This value will be
    //   interpolated with the other textureCoordinate values in the vertices that make up the
    //   triangle.
    out.textureCoordinate = vertexArray[vertexID].textureCoordinate;
    
    return out;
}

// The intermediate texture is already in destination texture orientation.
fragment float4 fragmentShader(RasterizerData in [[stage_in]],
                               texture2d<float, access::read> image [[texture(0)]]) {
    return image.read(uint2(in.clipSpacePosition.xy));
}

struct SortItem {
    float4 color;
    float key;
    uint segment;
    uint originalIndex;
};

bool lessThan(SortItem a, SortItem b) {
    if (a.segment != b.segment) return a.segment < b.segment;
    if (a.key != b.key) return a.key < b.key;
    return a.originalIndex < b.originalIndex;
}

kernel void pixelSort(texture2d<float, access::read> source [[texture(0)]],
                      texture2d<float, access::write> destination [[texture(1)]],
                      constant PixelSortUniforms &u [[buffer(0)]],
                      uint lane [[thread_index_in_threadgroup]],
                      uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup SortItem items[256];
    threadgroup uint barriers[256];
    const uint length = uint(u.configuration.z);
    const bool vertical = u.configuration.x != 0;
    int axis = (u.dispatchInfo.y + int(group.x)) * int(length) + int(lane);
    int cross = u.dispatchInfo.z + int(group.y);
    int2 point = vertical ? int2(cross, axis) : int2(axis, cross);
    int2 local = point - u.sourceRect.xy;
    bool valid = all(local >= int2(0)) && all(local < u.sourceRect.zw);
    if (u.configuration.w != 0) local.y = u.sourceRect.w - 1 - local.y;
    float4 original = valid ? source.read(uint2(local)) : float4(0);
    float luminance = original.a > 0 ? dot(original.rgb / original.a, float3(0.2126, 0.7152, 0.0722)) : 0;
    bool eligible = valid && original.a > 0 && all(isfinite(original)) && isfinite(luminance)
                    && luminance >= u.controls.x && luminance <= u.controls.y;
    barriers[lane] = eligible ? 0 : lane + 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Prefix maximum gives each contiguous eligible run its own segment.
    for (uint offset = 1; offset < length; offset <<= 1) {
        uint preceding = lane >= offset ? barriers[lane - offset] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        barriers[lane] = max(barriers[lane], preceding);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    SortItem item;
    item.color = original;
    item.key = eligible ? (u.configuration.y != 0 ? -luminance : luminance) : 0;
    item.segment = eligible ? 2 * barriers[lane] : 2 * lane + 1;
    item.originalIndex = lane;
    items[lane] = item;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Stable bitonic sort: segment first, luminance second, original index last.
    for (uint size = 2; size <= length; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            SortItem current = items[lane];
            SortItem partner = items[lane ^ stride];
            bool ascending = (lane & size) == 0;
            bool lowerLane = (lane & stride) == 0;
            bool chooseMin = ascending == lowerLane;
            bool partnerLess = lessThan(partner, current);
            SortItem selected = (chooseMin ? partnerLess : lessThan(current, partner)) ? partner : current;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            items[lane] = selected;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    int2 output = point - u.destinationRect.xy;
    if (all(output >= int2(0)) && all(output < u.destinationRect.zw)) {
        if (u.dispatchInfo.x != 0) output.y = u.destinationRect.w - 1 - output.y;
        float4 result = u.controls.z <= 0 ? original : (u.controls.z >= 1 ? items[lane].color : mix(original, items[lane].color, u.controls.z));
        destination.write(result, uint2(output));
    }
}

// Full-line sorting stores keys and source indices, not image colors, in scratch memory.
// Each line is sorted in 256-pixel groups, then stable parallel merges remove those boundaries.
struct FullSortItem {
    float key;
    uint segment;
    uint originalIndex;
    uint unused;
};

bool fullLess(FullSortItem a, FullSortItem b) {
    if (a.segment != b.segment) return a.segment < b.segment;
    if (a.key != b.key) return a.key < b.key;
    return a.originalIndex < b.originalIndex;
}

float4 fullSource(texture2d<float, access::read> source,
                  constant PixelSortUniforms &u, int axis, int cross) {
    int2 point = u.configuration.x != 0 ? int2(cross, axis) : int2(axis, cross);
    int2 local = point - u.sourceRect.xy;
    if (u.configuration.w != 0) local.y = u.sourceRect.w - 1 - local.y;
    return source.read(uint2(local));
}

// info: actual axis length, padded axis length, merge run length, batch line count.
kernel void fullSortInitialize(texture2d<float, access::read> source [[texture(0)]],
                               device FullSortItem *items [[buffer(0)]],
                               constant PixelSortUniforms &u [[buffer(1)]],
                               constant uint4 &info [[buffer(2)]],
                               uint line [[thread_position_in_grid]]) {
    if (line >= info.w) return;
    uint lastBarrier = 0;
    for (uint axis = 0; axis < info.y; ++axis) {
        FullSortItem item;
        item.originalIndex = axis;
        item.unused = 0;
        item.key = 0;
        item.segment = 0xffffffffu;
        if (axis < info.x) {
            float4 color = fullSource(source, u, int(axis), u.dispatchInfo.z + int(line));
            float luminance = color.a > 0 ? dot(color.rgb / color.a, float3(0.2126, 0.7152, 0.0722)) : 0;
            bool eligible = color.a > 0 && all(isfinite(color)) && isfinite(luminance)
                            && luminance >= u.controls.x && luminance <= u.controls.y;
            if (eligible) {
                item.segment = 2 * lastBarrier;
                item.key = u.configuration.y != 0 ? -luminance : luminance;
            } else {
                item.segment = 2 * axis + 1;
                lastBarrier = axis + 1;
            }
        }
        items[line * info.y + axis] = item;
    }
}

kernel void fullSortBlocks(device FullSortItem *items [[buffer(0)]],
                           constant uint4 &info [[buffer(1)]],
                           uint lane [[thread_index_in_threadgroup]],
                           uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup FullSortItem local[256];
    uint index = group.y * info.y + group.x * 256 + lane;
    local[lane] = items[index];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint size = 2; size <= 256; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            FullSortItem a = local[lane], b = local[lane ^ stride];
            bool chooseMin = ((lane & size) == 0) == ((lane & stride) == 0);
            FullSortItem selected = (chooseMin ? fullLess(b, a) : fullLess(a, b)) ? b : a;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            local[lane] = selected;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    items[index] = local[lane];
}

kernel void fullSortMerge(const device FullSortItem *input [[buffer(0)]],
                          device FullSortItem *output [[buffer(1)]],
                          constant uint4 &info [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
    if (index >= info.y * info.w) return;
    uint lineBase = (index / info.y) * info.y;
    uint axis = index % info.y;
    uint pairBase = (axis / (2 * info.z)) * (2 * info.z);
    uint runBase = (axis / info.z) * info.z;
    uint otherBase = runBase == pairBase ? pairBase + info.z : pairBase;
    uint low = min(otherBase, info.y), high = min(otherBase + info.z, info.y);
    FullSortItem item = input[index];
    // The original index makes every key unique, including equal-luminance pixels.
    while (low < high) {
        uint middle = low + (high - low) / 2;
        if (fullLess(input[lineBase + middle], item)) low = middle + 1;
        else high = middle;
    }
    uint rank = low - min(otherBase, info.y);
    output[lineBase + pairBase + (axis - runBase) + rank] = item;
}

kernel void fullSortOutput(texture2d<float, access::read> source [[texture(0)]],
                           texture2d<float, access::write> destination [[texture(1)]],
                           const device FullSortItem *items [[buffer(0)]],
                           constant PixelSortUniforms &u [[buffer(1)]],
                           constant uint4 &info [[buffer(2)]],
                           uint2 position [[thread_position_in_grid]]) {
    bool vertical = u.configuration.x != 0;
    uint count = uint(vertical ? u.destinationRect.w : u.destinationRect.z);
    if (position.x >= count || position.y >= info.w) return;
    int axis = int(position.x) + (vertical ? u.destinationRect.y : u.destinationRect.x);
    int cross = u.dispatchInfo.z + int(position.y);
    uint sortedAxis = items[position.y * info.y + uint(axis)].originalIndex;
    float4 original = fullSource(source, u, axis, cross);
    float4 sorted = fullSource(source, u, int(sortedAxis), cross);
    float4 result = u.controls.z <= 0 ? original : (u.controls.z >= 1 ? sorted : mix(original, sorted, u.controls.z));
    int2 point = vertical ? int2(cross, axis) : int2(axis, cross);
    int2 output = point - u.destinationRect.xy;
    if (u.dispatchInfo.x != 0) output.y = u.destinationRect.w - 1 - output.y;
    destination.write(result, uint2(output));
}
