// Host integration for a decoded-frame datamosh approximation. The host supplies current,
// previous, and delayed frames; no mutable feedback history depends on playback order.
import Foundation
import Metal

/// Preserves rational media times in the serializable render snapshot, including nonzero epochs.
private struct MoshTime: Codable {
    var value: Int64
    var scale: Int32
    var epoch: Int64
    init(_ time: CMTime) { value = time.value; scale = time.timescale; epoch = time.epoch }
    var time: CMTime { CMTime(value: value, timescale: scale, flags: .valid, epoch: epoch) }
}

/// Carries keyed controls and the exact historical requests from scheduling through rendering.
private struct MoshState: Codable {
    var amount = 0.7
    var block = 32.0
    var motion = 0.5
    var historyFrames = 6.0
    var seed = 1.0
    var mix = 1.0
    var previous = MoshTime(.zero)
    var history = MoshTime(.zero)
}

@objc(DatamoshPlugIn) final class DatamoshPlugIn: PixelSortPlugIn {
    /// Creates temporal block controls with their own IDs, separate from the other filters.
    override func addParameters() throws {
        guard let api = apiManager.api(for: FxParameterCreationAPI_v5.self) as? FxParameterCreationAPI_v5 else { throw pixelSortError("Datamosh controls are unavailable.") }
        let flags = FxParameterFlags(kFxParameterFlag_DEFAULT)
        func slider(_ name: String, _ id: UInt32, _ value: Double, _ low: Double = 0, _ high: Double = 1, _ step: Double = 0.01) -> Bool {
            api.addFloatSlider(withName: name, parameterID: id, defaultValue: value, parameterMin: low, parameterMax: high, sliderMin: low, sliderMax: high, delta: step, parameterFlags: flags)
        }
        let results = [slider("Amount", 30, 0.7), slider("History (frames)", 31, 6, 1, 60, 1),
                       slider("Block Size", 32, 32, 8, 128, 1), slider("Motion", 33, 0.5),
                       slider("Seed", 34, 1, 0, 10000, 1), slider("Mix", 35, 1)]
        guard results.allSatisfy({ $0 }) else { throw pixelSortError("Unable to create Datamosh controls.") }
    }

    /// Announces time remapping because this filter requests past input frames.
    override func properties(_ properties: AutoreleasingUnsafeMutablePointer<NSDictionary>?) throws {
        properties?.pointee = [kFxPropertyKey_MayRemapTime: true,
                              kFxPropertyKey_PixelTransformSupport: kFxPixelTransform_ScaleTranslate,
                              kFxPropertyKey_VariesWhenParamsAreStatic: true] as NSDictionary
    }

    /// Resolves keyed controls and clamps history to the start of the input before scheduling.
    override func pluginState(_ pluginState: AutoreleasingUnsafeMutablePointer<NSData>?, at renderTime: CMTime, quality qualityLevel: UInt) throws {
        guard let api = apiManager.api(for: FxParameterRetrievalAPI_v6.self) as? FxParameterRetrievalAPI_v6,
              let timing = apiManager.api(for: FxTimingAPI_v4.self) as? FxTimingAPI_v4 else {
            throw pixelSortError("Datamosh requires parameter and timing APIs.")
        }
        var s = MoshState()
        let ok = [api.getFloatValue(&s.amount, fromParameter: 30, at: renderTime),
                  api.getFloatValue(&s.historyFrames, fromParameter: 31, at: renderTime),
                  api.getFloatValue(&s.block, fromParameter: 32, at: renderTime),
                  api.getFloatValue(&s.motion, fromParameter: 33, at: renderTime),
                  api.getFloatValue(&s.seed, fromParameter: 34, at: renderTime),
                  api.getFloatValue(&s.mix, fromParameter: 35, at: renderTime)]
        guard ok.allSatisfy({ $0 }) else { throw pixelSortError("Unable to read Datamosh controls.") }
        func clamp(_ x: Double, _ low: Double, _ high: Double) -> Double { x.isFinite ? min(high, max(low, x)) : low }
        s.amount = clamp(s.amount,0,1); s.motion = clamp(s.motion,0,1); s.mix = clamp(s.mix,0,1)
        s.block = clamp(s.block,8,128).rounded(); s.seed = clamp(s.seed,0,10000).rounded()
        s.historyFrames = clamp(s.historyFrames,1,60).rounded()
        var duration = CMTime.invalid, start = CMTime.invalid
        timing.frameDuration(&duration)
        timing.startTimeOfInput(toFilter: &start)
        let times = try DatamoshTiming.past(render: renderTime, start: start, duration: duration, frames: Int32(s.historyFrames))
        s.previous = MoshTime(times.previous); s.history = MoshTime(times.history)
        pluginState?.pointee = try JSONEncoder().encode(s) as NSData
    }

    /// Decodes host-owned immutable state without fetching API values from a render thread.
    private func state(_ data: Data?) throws -> MoshState {
        guard let data = data else { throw pixelSortError("Missing Datamosh state.") }
        return try JSONDecoder().decode(MoshState.self,from:data)
    }

    /// Requests source frames with upstream filters included, deduplicating at clip boundaries.
    /// Output history is intentionally not requested, avoiding recursive render dependencies.
    func scheduleInputs(_ inputImageRequests: AutoreleasingUnsafeMutablePointer<NSArray?>?, withPluginState pluginState: Data?, at renderTime: CMTime) throws {
        let s = try state(pluginState)
        let times = DatamoshTiming.requests(current: renderTime, previous: s.previous.time,
                                            history: s.history.time, enabled: s.amount > 0 && s.mix > 0)
        let requests = try times.map { time -> FxImageTileRequest in
            guard let request = FxImageTileRequest(source: kFxImageTileRequestSourceEffectClip, time: time, includeFilters: true, parameterID: 0) else { throw pixelSortError("Unable to request Datamosh source frames.") }
            return request
        }
        inputImageRequests?.pointee = requests as NSArray
    }

    /// Expands all input tiles for sparse block matching and the largest displaced-history sample.
    override func sourceTileRect(_ sourceTileRect: UnsafeMutablePointer<FxRect>, sourceImageIndex: UInt, sourceImages: [FxImageTile], destinationTileRect: FxRect, destinationImage: FxImageTile, pluginState: Data?, at renderTime: CMTime) throws {
        guard Int(sourceImageIndex) < sourceImages.count else { throw pixelSortError("Missing Datamosh source descriptor.") }
        let s = try state(pluginState), image = sourceImages[Int(sourceImageIndex)].imagePixelBounds
        let search = max(1,Int32(s.block)/8) * 4
        let halo = s.amount > 0 && s.mix > 0 ? Int32(s.block) + search + Int32(ceil(Double(search)*s.motion*8)) + 1 : 0
        var rect = destinationTileRect
        rect.left = max(image.left,rect.left-halo); rect.right = min(image.right,rect.right+halo)
        rect.bottom = max(image.bottom,rect.bottom-halo); rect.top = min(image.top,rect.top+halo)
        sourceTileRect.pointee = rect
    }

    /// Uses explicit scheduled media times, falling back to the current image when history is missing.
    /// The GPU receives frame-specific rectangles and origins, so partial tiles need not align.
    override func renderDestinationImage(_ destinationImage: FxImageTile, sourceImages: [FxImageTile], pluginState: Data?, at renderTime: CMTime) throws {
        let s = try state(pluginState)
        guard let current = sourceImages.first(where:{ CMTimeCompare($0.mediaTime,renderTime) == 0 }) ?? sourceImages.first,
              current.requestError == nil,
              let device = MTLCopyAllDevices().first(where:{ $0.registryID == destinationImage.deviceRegistryID }),
              current.deviceRegistryID == device.registryID,
              let input = current.metalTexture(for:device), let output = destinationImage.metalTexture(for:device) else {
            throw pixelSortError("Datamosh current frame is unavailable.")
        }
        let image = current.imagePixelBounds
        func historical(_ time: CMTime) -> FxImageTile {
            sourceImages.first(where:{
                CMTimeCompare($0.mediaTime,time) == 0 && $0.requestError == nil && $0.deviceRegistryID == device.registryID &&
                $0.imagePixelBounds.right-$0.imagePixelBounds.left == image.right-image.left &&
                $0.imagePixelBounds.top-$0.imagePixelBounds.bottom == image.top-image.bottom
            }) ?? current
        }
        var previous = historical(s.previous.time), history = historical(s.history.time)
        if previous.ioSurface == nil { previous = current }
        if history.ioSurface == nil { history = current }
        // Texture fallbacks must also replace the metadata used to interpret those textures.
        var prior = input, past = input
        if let texture = previous.metalTexture(for:device) { prior = texture } else { previous = current }
        if let texture = history.metalTexture(for:device) { past = texture } else { history = current }
        let gpu = try MetalDeviceCache.deviceCache.gpu(registryID:device.registryID,format:output.pixelFormat)
        let bounds = destinationImage.tilePixelBounds
        let width = Int(bounds.right-bounds.left), height = Int(bounds.top-bounds.bottom)
        guard width > 0 && height > 0 else { return }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rgba32Float,width:width,height:height,mipmapped:false)
        descriptor.storageMode = .private; descriptor.usage = [.shaderRead,.shaderWrite]
        guard let intermediate = device.makeTexture(descriptor:descriptor), let command = gpu.queue.makeCommandBuffer() else { throw pixelSortError("Unable to allocate Datamosh output.") }
        func relative(_ r: FxRect, _ origin: FxRect) -> SIMD4<Int32> { SIMD4(r.left-origin.left,r.bottom-origin.bottom,r.right-r.left,r.top-r.bottom) }
        var u = DatamoshUniforms()
        u.currentRect = relative(current.tilePixelBounds,image)
        u.previousRect = relative(previous.tilePixelBounds,previous.imagePixelBounds)
        u.historyRect = relative(history.tilePixelBounds,history.imagePixelBounds)
        u.destinationRect = relative(bounds,image)
        let block = Int32(s.block)
        u.imageInfo = SIMD4(image.right-image.left,image.top-image.bottom,block,max(1,block/8))
        u.origins = SIMD4(current.imageOrigin == kFxImageOrigin_TOP_LEFT ? 1 : 0,previous.imageOrigin == kFxImageOrigin_TOP_LEFT ? 1 : 0,history.imageOrigin == kFxImageOrigin_TOP_LEFT ? 1 : 0,destinationImage.imageOrigin == kFxImageOrigin_TOP_LEFT ? 1 : 0)
        let firstX = u.destinationRect.x/block, firstY = u.destinationRect.y/block
        u.grid = SIMD4(firstX,firstY,(u.destinationRect.x+u.destinationRect.z+block-1)/block-firstX,(u.destinationRect.y+u.destinationRect.w+block-1)/block-firstY)
        // No historical input means a true current-frame passthrough, not spatial corruption.
        u.controls = SIMD4(history === current ? 0 : Float(s.amount),Float(s.motion),Float(s.mix),0)
        u.random = SIMD4(UInt32(s.seed),0,0,0)
        try gpu.encodeDatamosh(command:command,current:input,previous:prior,history:past,destination:intermediate,uniforms:u)
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
        if command.status != .completed { throw command.error ?? pixelSortError("Datamosh GPU rendering failed.") }
    }
}
