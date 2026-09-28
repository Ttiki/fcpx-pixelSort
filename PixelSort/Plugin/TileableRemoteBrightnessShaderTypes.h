//
//  TileableRemoteBrightnessShaderTypes.h
//  PixelSort
//
//  Created by Clément Combier on 28/09/2026.
//

#ifndef TileableRemoteBrightnessShaderTypes_h
#define TileableRemoteBrightnessShaderTypes_h

#import <simd/simd.h>

typedef enum BrightnessVertexInputIndex {
    BVI_Vertices        = 0,
    BVI_ViewportSize    = 1
} BrightnessVertexInputIndex;

typedef enum BrightnessTextureIndex {
    BTI_InputImage  = 0
} BrightnessTextureIndex;

typedef enum BrightnessFragmentIndex {
    BFI_Brightness  = 0
} BrightnessFragmentIndex;

typedef struct Vertex2D {
    vector_float2   position;
    vector_float2   textureCoordinate;
} Vertex2D;


typedef struct PixelSortUniforms {
    vector_int4 sourceRect; // image-relative left, bottom, width, height
    vector_int4 destinationRect;
    vector_int4 configuration; // vertical, reverse, block length, source top origin
    vector_int4 dispatchInfo; // destination top origin, first block, first cross-axis pixel, unused
    vector_float4 controls; // lower threshold, upper threshold, mix, unused
} PixelSortUniforms;

#endif /* TileableRemoteBrightnessShaderTypes_h */
