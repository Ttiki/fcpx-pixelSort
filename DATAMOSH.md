# Datamosh — 1V2T Glitch

Datamosh is a decoded-frame approximation of compression glitches. It estimates coarse backward motion between the current and previous input frames, then substitutes delayed-frame pixels in a seeded set of blocks. It does not edit codec motion vectors, delete I-frames, or maintain recursively accumulated output feedback.

## Try it in Motion

1. Quit Motion and Final Cut Pro.
2. Open `1V2T_Glitch.xcodeproj` and run **Wrapper Application** on **My Mac**.
3. Reopen Motion and add **Datamosh** from **Filters → 1V2T Glitch** to moving video.
4. Start with Amount **0.7**, History **6**, Block Size **32**, Motion **0.5**, Seed **1**, Mix **1**.

Moving subjects and cuts make the delayed blocks visible. A still frame with identical history can look unchanged. The first input frame has no earlier content and passes through.

## Controls

- **Amount:** proportion of blocks replaced with delayed content. Zero bypasses the effect.
- **History (frames):** delay of 1–60 native source-frame durations reported by the host. Larger values reuse older imagery. Retimed clips need host verification because the SDK reports native frame duration.
- **Block Size:** 8–128 render pixels. Larger blocks give a coarser breakup.
- **Motion:** exaggerates detected backward movement from 0 to 8 times the estimated vector. At zero, selected blocks display delayed pixels without warping.
- **Seed:** changes which blocks are selected. Selection remains stable for the same seed, independent of rendering order.
- **Mix:** blends replaced blocks with the current frame. Zero reproduces the current input.

## Implementation

- `1V2T_Glitch/Plugin/DatamoshPlugIn.swift` creates controls, snapshots keyframes, requests current/previous/history source frames, and maps their tiles to the GPU. Missing or incompatible historical frames fall back to current-frame passthrough. Upstream filters are included; the filter never recursively requests its own output.
- `DatamoshTiming.swift` computes rational sample times, clamps history to input start, and deduplicates requests. It can be tested without a running host.
- `Datamosh.metal` estimates motion with a sparse 3×3 patch over a 9×9 displacement grid, then applies delayed-frame block replacement. This is a deliberately coarse block matcher, not dense optical flow.
- `MetalDeviceCache.swift` compiles the two kernels once per GPU/output format and gives each render its own vector buffer. Temporal imagery comes from explicit host requests rather than an order-sensitive frame cache.

Source halos cover motion estimation and displaced historical samples so tile boundaries do not define the effect. Output dimensions stay equal to the input; sampling beyond its edges produces transparency. RGBA floats preserve alpha and HDR values through the intermediate.

## Verification and limits

The local Xcode build passes. Automated checks include 288 GPU/reference cases for motion, delayed content, alpha/HDR, seeds, Mix, tile boundaries, and differing image origins. Additional assertions check exact delayed-frame output with Motion zero and deterministic replay. The timing helper passes 48 rational-time scenarios plus invalid-time and pre-start cases.

Run both new test suites after building:

```sh
bash Scripts/test-datamosh.sh '/path/to/PixelSort XPC Service.pluginkit/Contents/Resources/default.metallib'
```

The embedded service retains its old product name to preserve registration; the visible application/project is named 1V2T Glitch. Xcode's stale source and entitlement paths were updated to the renamed folder.

Motion/FCP discovery, temporal input delivery, retimed footage, long clips, and real-time performance remain unverified in the host application. This version does not ship a published Final Cut Effect template. Validate in Motion before publishing one.
