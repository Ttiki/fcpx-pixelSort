// PixelSort's host-facing controller. It creates controls, snapshots keyframed values,
// requests the pixels each tile needs, and schedules brightness sorting or text streaking.
// Foundation supplies Codable/Data and host errors; FxPlug/Metal types arrive via the bridging header.

import Foundation

/// Carries one render-time snapshot without sharing mutable UI values across render threads.
/// Codable serializes the snapshot to the opaque data FxPlug passes back during rendering.
private struct SortState: Codable {
    var lower: Double = 0.15
    var upper: Double = 0.85
    var mix: Double = 1
    var vertical = false
    var reverse = false
    var amount: Double = 0.25
    var mode: Int32 = 0
    var seed: UInt32 = 1
    /// Converts the normalized length slider to render pixels on the selected axis.
    /// A one-pixel result selects the copy path; the image dimension is the upper bound.
    func length(for axis: Int32) -> Int32 {
        let value = amount.isFinite ? max(0, min(1, amount)) : 0
        return max(1, Int32((Double(axis) * value).rounded()))
    }
}

/// Exposes the stable Objective-C class name registered in Info.plist.
/// FxTileableEffect callbacks connect the host image pipeline to the shared GPU implementation.
@objc(PixelSortPlugIn) class PixelSortPlugIn: NSObject, FxTileableEffect {
    let apiManager: PROAPIAccessing
    /// Retains the host API manager so callbacks can create controls and retrieve keyed values.
    required init?(apiManager: PROAPIAccessing) { self.apiManager = apiManager }

    /// Creates the filter inspector and assigns persistent IDs to its controls.
    /// New mode/seed IDs leave existing direction, threshold, mix, and length animations intact.
    func addParameters() throws {
        guard let api = apiManager.api(for: FxParameterCreationAPI_v5.self) as? FxParameterCreationAPI_v5 else {
            throw pixelSortError("Parameter creation API is unavailable.")
        }
        let flags = FxParameterFlags(kFxParameterFlag_DEFAULT)
        // IDs start at 10 so the starter's Brightness animation cannot become a different control.
        let results = [
            api.addPopupMenu(withName: "Mode", parameterID: 17, defaultValue: 0, menuEntries: ["Brightness Sort", "Text Streaks"], parameterFlags: flags),
            api.addFloatSlider(withName: "Streak Seed", parameterID: 18, defaultValue: 1, parameterMin: 0, parameterMax: 10000, sliderMin: 0, sliderMax: 10000, delta: 1, parameterFlags: flags),
            api.addPopupMenu(withName: "Direction", parameterID: 10, defaultValue: 0, menuEntries: ["Horizontal", "Vertical"], parameterFlags: flags),
            api.addFloatSlider(withName: "Lower Brightness", parameterID: 11, defaultValue: 0.15, parameterMin: 0, parameterMax: 16, sliderMin: 0, sliderMax: 1, delta: 0.01, parameterFlags: flags),
            api.addFloatSlider(withName: "Upper Brightness", parameterID: 12, defaultValue: 0.85, parameterMin: 0, parameterMax: 16, sliderMin: 0, sliderMax: 1, delta: 0.01, parameterFlags: flags),
            api.addToggleButton(withName: "Reverse", parameterID: 13, defaultValue: false, parameterFlags: flags),
            api.addFloatSlider(withName: "Mix", parameterID: 14, defaultValue: 1, parameterMin: 0, parameterMax: 1, sliderMin: 0, sliderMax: 1, delta: 0.01, parameterFlags: flags),
            api.addFloatSlider(withName: "Sort Length", parameterID: 16, defaultValue: 0.25, parameterMin: 0, parameterMax: 1, sliderMin: 0, sliderMax: 1, delta: 0.001, parameterFlags: flags)
        ]
        guard results.allSatisfy({ $0 }) else { throw pixelSortError("Unable to create PixelSort controls.") }
    }

    /// Declares render behavior to the host. Neither sorting nor seeded streaks uses wall-clock time,
    /// so unchanged parameters may safely reuse a cached frame.
    func properties(_ properties: AutoreleasingUnsafeMutablePointer<NSDictionary>?) throws {
        properties?.pointee = [
            kFxPropertyKey_MayRemapTime: false,
            kFxPropertyKey_PixelTransformSupport: kFxPixelTransform_ScaleTranslate,
            kFxPropertyKey_VariesWhenParamsAreStatic: false
        ] as NSDictionary
    }

    /// Reads all controls at the requested timeline time, then serializes an immutable snapshot.
    /// The GPU render callback consumes this snapshot instead of querying the host UI APIs.
    func pluginState(_ pluginState: AutoreleasingUnsafeMutablePointer<NSData>?, at renderTime: CMTime, quality qualityLevel: UInt) throws {
        guard let api = apiManager.api(for: FxParameterRetrievalAPI_v6.self) as? FxParameterRetrievalAPI_v6 else {
            throw pixelSortError("Parameter retrieval API is unavailable.")
        }
        var state = SortState()
        var direction: Int32 = 0
        var reverse = ObjCBool(false)
        var seed = 1.0
        let results = [
            api.getIntValue(&state.mode, fromParameter: 17, at: renderTime),
            api.getFloatValue(&seed, fromParameter: 18, at: renderTime),
            api.getIntValue(&direction, fromParameter: 10, at: renderTime),
            api.getFloatValue(&state.lower, fromParameter: 11, at: renderTime),
            api.getFloatValue(&state.upper, fromParameter: 12, at: renderTime),
            api.getBoolValue(&reverse, fromParameter: 13, at: renderTime),
            api.getFloatValue(&state.mix, fromParameter: 14, at: renderTime),
            api.getFloatValue(&state.amount, fromParameter: 16, at: renderTime)
        ]
        guard results.allSatisfy({ $0 }) else { throw pixelSortError("Unable to read PixelSort controls.") }
        state.vertical = direction == 1
        state.reverse = reverse.boolValue
        state.mode = state.mode == 1 ? 1 : 0
        state.seed = UInt32(seed.isFinite ? min(10000, max(0, seed)).rounded() : 1)
        pluginState?.pointee = try JSONEncoder().encode(state) as NSData
    }

    /// Decodes the exact snapshot associated with this render; missing data becomes a host error.
    private func state(from data: Data?) throws -> SortState {
        guard let data = data else { throw pixelSortError("Missing render state.") }
        return try JSONDecoder().decode(SortState.self, from: data)
    }

    /// Keeps the output canvas equal to the input canvas. Effects can fill transparent space inside
    /// these bounds; they do not grow the layer beyond its original dimensions.
    func destinationImageRect(_ destinationImageRect: UnsafeMutablePointer<FxRect>, sourceImages: [FxImageTile], destinationImage: FxImageTile, pluginState: Data?, at renderTime: CMTime) throws {
        guard let source = sourceImages.first else { throw pixelSortError("Missing source image.") }
        destinationImageRect.pointee = source.imagePixelBounds
    }

    /// Requests complete rows or columns when neighboring pixels can affect the current tile.
    /// The cross-axis extent stays narrow so the host can still split the frame into tiles.
    func sourceTileRect(_ sourceTileRect: UnsafeMutablePointer<FxRect>, sourceImageIndex: UInt, sourceImages: [FxImageTile], destinationTileRect: FxRect, destinationImage: FxImageTile, pluginState: Data?, at renderTime: CMTime) throws {
        guard let source = sourceImages.first else { throw pixelSortError("Missing source image.") }
        let settings = try state(from: pluginState)
        let image = source.imagePixelBounds
        var rect = destinationTileRect
        let axis = settings.vertical ? image.top - image.bottom : image.right - image.left
        if settings.length(for: axis) > 1 && settings.mix > 0 {
            // Both merging and streak scans need pixels before the tile; local input would create seams.
            if settings.vertical { rect.bottom = image.bottom; rect.top = image.top }
            else { rect.left = image.left; rect.right = image.right }
        }
        rect.left = max(image.left, rect.left); rect.right = min(image.right, rect.right)
        rect.bottom = max(image.bottom, rect.bottom); rect.top = min(image.top, rect.top)
        sourceTileRect.pointee = rect
    }

    /// Renders one output tile using textures on the GPU selected by the host.
    /// It computes into an RGBA-float intermediate, then draws into the host texture format.
    func renderDestinationImage(_ destinationImage: FxImageTile, sourceImages: [FxImageTile], pluginState: Data?, at renderTime: CMTime) throws {
        guard let source = sourceImages.first,
              source.deviceRegistryID == destinationImage.deviceRegistryID,
              let device = MTLCopyAllDevices().first(where: { $0.registryID == destinationImage.deviceRegistryID }),
              let input = source.metalTexture(for: device),
              let output = destinationImage.metalTexture(for: device) else {
            throw pixelSortError("Unable to access the host's image textures on a shared GPU.")
        }
        let settings = try state(from: pluginState)
        let gpu = try MetalDeviceCache.deviceCache.gpu(registryID: device.registryID, format: output.pixelFormat)
        let bounds = destinationImage.tilePixelBounds
        let width = Int(bounds.right - bounds.left), height = Int(bounds.top - bounds.bottom)
        guard width > 0, height > 0 else { return }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        // The intermediate remains GPU-only; floating-point storage preserves HDR channel values.
        descriptor.storageMode = .private
        descriptor.usage = [.shaderRead, .shaderWrite]
        guard let intermediate = device.makeTexture(descriptor: descriptor),
              let command = gpu.queue.makeCommandBuffer() else {
            throw pixelSortError("Unable to allocate PixelSort rendering resources.")
        }
        let image = source.imagePixelBounds
        /// Expresses tile bounds in one image-relative coordinate system, independent of tile origin.
        func relative(_ rect: FxRect) -> SIMD4<Int32> {
            SIMD4(rect.left - image.left, rect.bottom - image.bottom, rect.right - rect.left, rect.top - rect.bottom)
        }
        let dest = relative(bounds)
        let axisLength = settings.vertical ? image.top - image.bottom : image.right - image.left
        let length = settings.length(for: axisLength)
        var uniforms = PixelSortUniforms()
        uniforms.sourceRect = relative(source.tilePixelBounds)
        uniforms.destinationRect = dest
        uniforms.configuration = SIMD4(settings.vertical ? 1 : 0, settings.reverse ? 1 : 0, length == axisLength ? 0 : length, source.imageOrigin == kFxImageOrigin_TOP_LEFT ? 1 : 0)
        uniforms.dispatchInfo = SIMD4(destinationImage.imageOrigin == kFxImageOrigin_TOP_LEFT ? 1 : 0, 0, settings.vertical ? dest.x : dest.y, 0)
        uniforms.controls = SIMD4(Float(settings.lower), Float(settings.upper), Float(settings.mix), 0)
        // Bypass avoids unnecessary sorting and makes zero length/mix reproduce the original.
        if length <= 1 || settings.mix <= 0 {
            try gpu.encodeCopy(command: command, source: input, destination: intermediate, uniforms: uniforms)
        } else if settings.mode == 1 {
            try gpu.encodeTextStreaks(command: command, source: input, destination: intermediate,
                                      uniforms: uniforms, axisLength: Int(axisLength), seed: settings.seed)
        } else {
            try gpu.encodeFullSort(command: command, source: input, destination: intermediate,
                                   uniforms: uniforms, axisLength: Int(axisLength))
        }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = output
        pass.colorAttachments[0].loadAction = .dontCare
        // Explicitly retain the rendered image for the host after the pass finishes.
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
        // FxPlug must not consume the output before GPU writes finish.
        command.waitUntilCompleted()
        if command.status != .completed { throw command.error ?? pixelSortError("PixelSort GPU rendering failed.") }
    }
}
