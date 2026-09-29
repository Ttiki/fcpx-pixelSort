// Host controllers for Signal Tear and RGB Split. A common base handles controls, timeline
// snapshots, source neighborhoods, and output rendering; subclasses choose the effect kernel branch.
// Foundation handles snapshots/math, and Metal provides the GPU command and texture interfaces.

import Foundation
import Metal

/// Stores normalized controls and a deterministic time bucket for one requested frame.
private struct GlitchState: Codable {
    var amount = 0.3
    var scale = 0.25
    var speed = 8.0
    var seed = 1.0
    var mix = 1.0
    var density = 0.45
    var angle = 0.0
    var tick: UInt32 = 0
}

// The original filter supplies image dimensions and the shared API manager.
// Every time-dependent callback is overridden; PixelSort itself remains unchanged.
/// Shares the FxPlug integration for both glitch filters while inheriting PixelSort canvas sizing.
/// Its own state and render overrides keep animated glitches separate from sorting behavior.
class GlitchPlugIn: PixelSortPlugIn {
    var effectKind: UInt32 { 0 }

    /// Builds shared Amount/Speed/Seed/Mix controls and the controls specific to each effect.
    override func addParameters() throws {
        guard let api = apiManager.api(for: FxParameterCreationAPI_v5.self) as? FxParameterCreationAPI_v5 else {
            throw pixelSortError("Glitch parameter creation API is unavailable.")
        }
        let flags = FxParameterFlags(kFxParameterFlag_DEFAULT)
        /// Creates a bounded numeric control with a consistent slider range and step size.
        func slider(_ name: String, _ id: UInt32, _ value: Double, _ lower: Double = 0, _ upper: Double = 1, _ step: Double = 0.01) -> Bool {
            api.addFloatSlider(withName: name, parameterID: id, defaultValue: value, parameterMin: lower, parameterMax: upper, sliderMin: lower, sliderMax: upper, delta: step, parameterFlags: flags)
        }
        var results = [slider("Amount", 10, effectKind == 0 ? 0.3 : 0.1),
                       slider("Speed", 12, effectKind == 0 ? 8 : 0, 0, 30, 0.1),
                       slider("Seed", 13, 1, 0, 10000, 1), slider("Mix", 14, 1)]
        if effectKind == 0 {
            results += [slider("Band Scale", 11, 0.25), slider("Density", 15, 0.45)]
        } else {
            results += [slider("Angle", 16, 0, -180, 180, 1)]
        }
        guard results.allSatisfy({ $0 }) else { throw pixelSortError("Unable to create glitch controls.") }
    }

    /// Marks the output as time-varying even when controls are static, so animation is not cached away.
    override func properties(_ properties: AutoreleasingUnsafeMutablePointer<NSDictionary>?) throws {
        properties?.pointee = [
            kFxPropertyKey_MayRemapTime: false,
            kFxPropertyKey_PixelTransformSupport: kFxPixelTransform_ScaleTranslate,
            kFxPropertyKey_VariesWhenParamsAreStatic: true
        ] as NSDictionary
    }

    /// Captures keyed controls and derives a repeatable random time bucket from timeline seconds.
    /// No persistent random generator is advanced, so seeking and exporting the same frame agree.
    override func pluginState(_ pluginState: AutoreleasingUnsafeMutablePointer<NSData>?, at renderTime: CMTime, quality qualityLevel: UInt) throws {
        guard let api = apiManager.api(for: FxParameterRetrievalAPI_v6.self) as? FxParameterRetrievalAPI_v6 else {
            throw pixelSortError("Glitch parameter retrieval API is unavailable.")
        }
        var state = GlitchState()
        var results = [api.getFloatValue(&state.amount, fromParameter: 10, at: renderTime),
                       api.getFloatValue(&state.speed, fromParameter: 12, at: renderTime),
                       api.getFloatValue(&state.seed, fromParameter: 13, at: renderTime),
                       api.getFloatValue(&state.mix, fromParameter: 14, at: renderTime)]
        if effectKind == 0 {
            results += [api.getFloatValue(&state.scale, fromParameter: 11, at: renderTime),
                        api.getFloatValue(&state.density, fromParameter: 15, at: renderTime)]
        } else {
            results += [api.getFloatValue(&state.angle, fromParameter: 16, at: renderTime)]
        }
        guard results.allSatisfy({ $0 }) else { throw pixelSortError("Unable to read glitch controls.") }
        /// Sanitizes values before floating-point values become integer seeds or GPU parameters.
        func clamp(_ x: Double, _ low: Double, _ high: Double) -> Double { x.isFinite ? min(high, max(low, x)) : low }
        state.amount = clamp(state.amount, 0, 1); state.mix = clamp(state.mix, 0, 1)
        state.scale = clamp(state.scale, 0, 1); state.density = clamp(state.density, 0, 1)
        state.speed = clamp(state.speed, 0, 30); state.seed = clamp(state.seed, 0, 10000).rounded()
        state.angle = clamp(state.angle, -180, 180)
        // Timeline time, never wall-clock time, determines the pattern. Speed zero freezes it.
        let seconds = CMTimeGetSeconds(renderTime)
        if state.speed > 0 && seconds.isFinite {
            let wrapped = floor(seconds * state.speed).truncatingRemainder(dividingBy: 4294967296)
            state.tick = UInt32(wrapped < 0 ? wrapped + 4294967296 : wrapped)
        }
        pluginState?.pointee = try JSONEncoder().encode(state) as NSData
    }

    /// Recovers the glitch snapshot passed by the host and rejects absent render state.
    private func state(from data: Data?) throws -> GlitchState {
        guard let data = data else { throw pixelSortError("Missing glitch render state.") }
        return try JSONDecoder().decode(GlitchState.self, from: data)
    }

    /// Expands the requested source neighborhood enough to cover the largest possible displacement.
    /// The extra pixel covers rounding at the edge; clipping keeps the request within source bounds.
    override func sourceTileRect(_ sourceTileRect: UnsafeMutablePointer<FxRect>, sourceImageIndex: UInt, sourceImages: [FxImageTile], destinationTileRect: FxRect, destinationImage: FxImageTile, pluginState: Data?, at renderTime: CMTime) throws {
        guard let source = sourceImages.first else { throw pixelSortError("Missing glitch input.") }
        let settings = try state(from: pluginState)
        let image = source.imagePixelBounds
        let width = Double(image.right - image.left)
        var rect = destinationTileRect
        if settings.amount > 0 && settings.mix > 0 {
            let radius = settings.amount * width * (effectKind == 0 ? 0.25 : 0.05)
            let radians = settings.angle * .pi / 180
            let x = Int32(ceil(effectKind == 0 ? radius : abs(cos(radians) * radius))) + 1
            let y = effectKind == 0 ? 0 : Int32(ceil(abs(sin(radians) * radius))) + 1
            rect.left -= x; rect.right += x; rect.bottom -= y; rect.top += y
        }
        rect.left = max(image.left, rect.left); rect.right = min(image.right, rect.right)
        rect.bottom = max(image.bottom, rect.bottom); rect.top = min(image.top, rect.top)
        sourceTileRect.pointee = rect
    }

    /// Maps a source tile into shared GPU uniforms, runs the selected glitch, and draws the result.
    /// A float intermediate supports the same compute kernels across the host output formats.
    override func renderDestinationImage(_ destinationImage: FxImageTile, sourceImages: [FxImageTile], pluginState: Data?, at renderTime: CMTime) throws {
        guard let source = sourceImages.first,
              source.deviceRegistryID == destinationImage.deviceRegistryID,
              let device = MTLCopyAllDevices().first(where: { $0.registryID == destinationImage.deviceRegistryID }),
              let input = source.metalTexture(for: device),
              let output = destinationImage.metalTexture(for: device) else {
            throw pixelSortError("Unable to access glitch image textures.")
        }
        let state = try state(from: pluginState)
        let gpu = try MetalDeviceCache.deviceCache.gpu(registryID: device.registryID, format: output.pixelFormat)
        let bounds = destinationImage.tilePixelBounds
        let width = Int(bounds.right - bounds.left), height = Int(bounds.top - bounds.bottom)
        guard width > 0 && height > 0 else { return }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.storageMode = .private; descriptor.usage = [.shaderRead, .shaderWrite]
        guard let intermediate = device.makeTexture(descriptor: descriptor), let command = gpu.queue.makeCommandBuffer() else {
            throw pixelSortError("Unable to allocate glitch resources.")
        }
        let image = source.imagePixelBounds
        /// Converts both source and output bounds to the same image-anchored pixel coordinates.
        func relative(_ r: FxRect) -> SIMD4<Int32> { SIMD4(r.left-image.left, r.bottom-image.bottom, r.right-r.left, r.top-r.bottom) }
        var u = GlitchUniforms()
        u.sourceRect = relative(source.tilePixelBounds); u.destinationRect = relative(bounds)
        u.imageInfo = SIMD4(image.right-image.left, image.top-image.bottom, source.imageOrigin == kFxImageOrigin_TOP_LEFT ? 1 : 0, destinationImage.imageOrigin == kFxImageOrigin_TOP_LEFT ? 1 : 0)
        u.configuration = SIMD4(effectKind, UInt32(state.seed), state.tick, state.speed > 0 ? 1 : 0)
        u.controls = SIMD4(Float(state.amount), Float(state.scale), Float(state.mix), Float(state.density))
        // Scale displacement to image width so the control behaves proportionally at different resolutions.
        let radius = state.amount * Double(image.right-image.left) * 0.05
        let angle = state.angle * .pi / 180
        u.offset = SIMD4(Float(cos(angle)*radius), Float(sin(angle)*radius), 0, 0)
        try gpu.encodeGlitch(command: command, source: input, destination: intermediate, uniforms: u)
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = output
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let render = command.makeRenderCommandEncoder(descriptor: pass) else {
            throw pixelSortError("Unable to create the output render encoder.")
        }
        var vertices = [
            Vertex2D(position: SIMD2(Float(width)/2, -Float(height)/2), textureCoordinate: SIMD2(1, 1)),
            Vertex2D(position: SIMD2(-Float(width)/2, -Float(height)/2), textureCoordinate: SIMD2(0, 1)),
            Vertex2D(position: SIMD2(Float(width)/2, Float(height)/2), textureCoordinate: SIMD2(1, 0)),
            Vertex2D(position: SIMD2(-Float(width)/2, Float(height)/2), textureCoordinate: SIMD2(0, 0))
        ]
        var viewport = SIMD2<UInt32>(UInt32(width), UInt32(height))
        render.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(width), height: Double(height), znear: -1, zfar: 1))
        render.setRenderPipelineState(gpu.render)
        render.setVertexBytes(&vertices, length: MemoryLayout<Vertex2D>.stride * 4, index: 0)
        render.setVertexBytes(&viewport, length: MemoryLayout.size(ofValue: viewport), index: 1)
        render.setFragmentTexture(intermediate, index: 0)
        render.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        render.endEncoding()
        command.commit()
        // The host owns the destination and may reuse it as soon as this callback returns.
        command.waitUntilCompleted()
        if command.status != .completed { throw command.error ?? pixelSortError("Glitch GPU rendering failed.") }
    }
}

/// Registers a distinct filter class whose shared shader branch displaces horizontal bands.
@objc(SignalTearPlugIn) final class SignalTearPlugIn: GlitchPlugIn {
    override var effectKind: UInt32 { 0 }
}

/// Registers a distinct filter class whose shared shader branch separates RGB channel positions.
@objc(RGBSplitPlugIn) final class RGBSplitPlugIn: GlitchPlugIn {
    override var effectKind: UInt32 { 1 }
}
