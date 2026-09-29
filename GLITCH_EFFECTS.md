# 1V2T Glitch — first effects

The PixelSort application registers **PixelSort**, **Signal Tear**, and **RGB Split** together in Motion's **1V2T Glitch** filter category. PixelSort retains its original plug-in UUID and existing parameter IDs, so moving categories does not create a different effect.

## Try the effects

1. Quit Motion and Final Cut Pro.
2. Open `PixelSort.xcodeproj` and run the **Wrapper Application** scheme on **My Mac**.
3. Reopen Motion. Add white text over a transparent background or a video clip.
4. Apply **Signal Tear** from **Library → Filters → 1V2T Glitch**.
5. Add **RGB Split** after it to combine the effects.

The application is still named PixelSort. Final Cut effect templates have not yet been published; verify these filters in Motion first, then publish their controls through Motion Final Cut Effect templates.

## PixelSort: brightness or text streaks

**Mode → Brightness Sort** retains the existing brightness-threshold sorting behavior.

**Mode → Text Streaks** extends foreground pixels into transparent space by a seeded random distance per row or column. It copies edge color and coverage rather than rearranging equal white pixels. Original foreground pixels remain intact; the most opaque color in a connected foreground run supplies its trail so antialiased edges do not make it faint.

- **Direction:** horizontal follows rows, vertical follows columns.
- **Reverse:** flips travel direction; default is rightward or downward in image coordinates.
- **Sort Length (0–1):** maximum randomized reach as a fraction of width/height; 0 bypasses, and 1 permits reach up to the full image dimension. Individual rows/columns still choose different lengths.
- **Streak Seed:** changes the static random pattern. The same seed and image coordinates repeat exactly. Keyframe the seed if you want the pattern to change over time.
- **Mix:** blends the original and streaked image.

Brightness thresholds affect Brightness Sort only. Streak Seed affects Text Streaks only. Text Streaks uses alpha, so use white or colored text on a **transparent** background; an opaque black background will not be treated as empty space. Near-zero alpha (at most 0.01) is treated as a gap. Add transparent padding around tightly cropped text: the output canvas still matches the input bounds.

Start with Mode Text Streaks, Sort Length 0.15, Streak Seed 7, and Mix 0.7. Freshly apply the filter after rebuilding so Motion exposes the new Mode and Streak Seed controls.

## Signal Tear

- **Amount (0–1):** maximum sideways displacement, reaching 25% of the input width at 1.
- **Band Scale (0–1):** horizontal band height, approximately 0.5–12.5% of input height.
- **Density (0–1):** proportion of bands eligible to shift.
- **Speed (0–30):** pattern changes per timeline second. Set 0 to freeze the pattern.
- **Seed:** selects a reproducible random pattern.
- **Mix (0–1):** blends the original and displaced images.

Start with Amount 0.15, Band Scale 0.2, Density 0.45, Speed 8, and Mix 1. Animate Amount for brief interruptions.

## RGB Split

- **Amount (0–1):** red/blue channel displacement, reaching 5% of the input width in each direction at 1. Green stays in place.
- **Angle (−180–180):** displacement direction; 0 is horizontal and 90 is vertical.
- **Speed (0–30):** changes displacement intensity. Set 0 for a static split.
- **Seed:** controls the animated intensity pattern; it has no visible effect while Speed is 0.
- **Mix (0–1):** blends the original and split images.

Start with Amount 0.05–0.1 and Speed 0. Alpha follows the shifted channel coverage, so colored edges remain visible on transparent text.

## Behavior and limitations

Randomness depends on seed, band position, and timeline time rather than wall-clock time. Rendering the same settings at the same time is repeatable. Keyframing Speed changes the time bucket directly; it does not integrate a smoothly varying animation speed.

Both effects preserve the input image dimensions. Samples beyond the image edges become transparent, and displaced content beyond the original bounds is clipped. Add transparent padding around tightly cropped text if you want room for larger displacement. Positions use whole render pixels for a sharp digital appearance. Preview/export comparisons at different render scales still need host testing.

## Verification

A local Xcode Debug build passes. The GPU test compares 288 cases against a CPU reference: both effects, zero and nonzero amounts, mixing, seeds, time, transparency, HDR values, differing texture origins, and cropped tiles. Additional assertions check white-text distortion, alpha fringes, deterministic replay, and animated changes. PixelSort's 192 GPU regression cases also pass.

To run the glitch checks after building:

```sh
bash Scripts/test-glitch.sh '/path/to/PixelSort XPC Service.pluginkit/Contents/Resources/default.metallib' '/tmp/glitch-preview.png'
```

This requires a Mac with a Metal GPU and Xcode's command-line tools. The shader checks do not establish Motion/Final Cut discovery, UI behavior, or real-time playback performance; those still need host testing.

### Text Streaks checks

```sh
bash Scripts/test-text-streaks.sh '/path/to/PixelSort XPC Service.pluginkit/Contents/Resources/default.metallib' '/tmp/text-streak-preview.png'
```

384 GPU/reference cases cover both directions and travel signs, random seeds, finite/full lengths, bypass, Mix, different texture origins, and partial output tiles. Additional checks verify preserved letters, visible extension of white text, and deterministic seed behavior. The CPU reference searches backward from each gap independently of the GPU’s streaming scan.

Source files use file-level purpose summaries, function-level purpose/approach comments, and selective explanations of decisions such as tile expansion, stable tie-breaking, padding, alpha handling, and GPU synchronization.
