import Foundation

private struct SortState: Codable {
    var lower: Double = 0.15
    var upper: Double = 0.85
    var mix: Double = 1
    var vertical = false
    var reverse = false
    var length: Int32 = 256
}

@objc(PixelSortPlugIn) class PixelSortPlugIn: NSObject, FxTileableEffect {
    let apiManager: PROAPIAccessing
    required init?(apiManager: PROAPIAccessing) { self.apiManager = apiManager }

    func addParameters() throws {
        guard let api = apiManager.api(for: FxParameterCreationAPI_v5.self) as? FxParameterCreationAPI_v5 else {
            throw pixelSortError("Parameter creation API is unavailable.")
        }
        let flags = FxParameterFlags(kFxParameterFlag_DEFAULT)
        // IDs start at 10 so the starter's Brightness animation cannot become a different control.
        let results = [
            api.addPopupMenu(withName: "Direction", parameterID: 10, defaultValue: 0, menuEntries: ["Horizontal", "Vertical"], parameterFlags: flags),
            api.addFloatSlider(withName: "Lower Brightness", parameterID: 11, defaultValue: 0.15, parameterMin: 0, parameterMax: 16, sliderMin: 0, sliderMax: 1, delta: 0.01, parameterFlags: flags),
            api.addFloatSlider(withName: "Upper Brightness", parameterID: 12, defaultValue: 0.85, parameterMin: 0, parameterMax: 16, sliderMin: 0, sliderMax: 1, delta: 0.01, parameterFlags: flags),
            api.addToggleButton(withName: "Reverse", parameterID: 13, defaultValue: false, parameterFlags: flags),
            api.addFloatSlider(withName: "Mix", parameterID: 14, defaultValue: 1, parameterMin: 0, parameterMax: 1, sliderMin: 0, sliderMax: 1, delta: 0.01, parameterFlags: flags),
            api.addPopupMenu(withName: "Sort Length", parameterID: 15, defaultValue: 2, menuEntries: ["64 pixels", "128 pixels", "256 pixels", "Full image"], parameterFlags: flags)
        ]
        guard results.allSatisfy({ $0 }) else { throw pixelSortError("Unable to create PixelSort controls.") }
    }

    func properties(_ properties: AutoreleasingUnsafeMutablePointer<NSDictionary>?) throws {
        properties?.pointee = [
            kFxPropertyKey_MayRemapTime: false,
            kFxPropertyKey_PixelTransformSupport: kFxPixelTransform_ScaleTranslate,
            kFxPropertyKey_VariesWhenParamsAreStatic: false
        ] as NSDictionary
    }

    func pluginState(_ pluginState: AutoreleasingUnsafeMutablePointer<NSData>?, at renderTime: CMTime, quality qualityLevel: UInt) throws {
        guard let api = apiManager.api(for: FxParameterRetrievalAPI_v6.self) as? FxParameterRetrievalAPI_v6 else {
            throw pixelSortError("Parameter retrieval API is unavailable.")
        }
        var state = SortState()
        var direction: Int32 = 0
        var length: Int32 = 2
        var reverse = ObjCBool(false)
        let results = [
            api.getIntValue(&direction, fromParameter: 10, at: renderTime),
            api.getFloatValue(&state.lower, fromParameter: 11, at: renderTime),
            api.getFloatValue(&state.upper, fromParameter: 12, at: renderTime),
            api.getBoolValue(&reverse, fromParameter: 13, at: renderTime),
            api.getFloatValue(&state.mix, fromParameter: 14, at: renderTime),
            api.getIntValue(&length, fromParameter: 15, at: renderTime)
        ]
        guard results.allSatisfy({ $0 }) else { throw pixelSortError("Unable to read PixelSort controls.") }
        state.vertical = direction == 1
        state.reverse = reverse.boolValue
        state.length = [64, 128, 256, 0][Int(max(0, min(3, length)))]
        pluginState?.pointee = try JSONEncoder().encode(state) as NSData
    }

    private func state(from data: Data?) throws -> SortState {
        guard let data = data else { throw pixelSortError("Missing render state.") }
        return try JSONDecoder().decode(SortState.self, from: data)
    }

    func destinationImageRect(_ destinationImageRect: UnsafeMutablePointer<FxRect>, sourceImages: [FxImageTile], destinationImage: FxImageTile, pluginState: Data?, at renderTime: CMTime) throws {
        guard let source = sourceImages.first else { throw pixelSortError("Missing source image.") }
        destinationImageRect.pointee = source.imagePixelBounds
    }

    func sourceTileRect(_ sourceTileRect: UnsafeMutablePointer<FxRect>, sourceImageIndex: UInt, sourceImages: [FxImageTile], destinationTileRect: FxRect, destinationImage: FxImageTile, pluginState: Data?, at renderTime: CMTime) throws {
        guard let source = sourceImages.first else { throw pixelSortError("Missing source image.") }
        let settings = try state(from: pluginState)
        let image = source.imagePixelBounds
        var rect = destinationTileRect
        let n = settings.length
        // Expand only along the sorting axis, to image-anchored block boundaries.
        if n == 0 {
            // Full lines are required even when the host requests only part of the output.
            // Keep the cross-axis range narrow so the host can still tile the image.
            if settings.vertical { rect.bottom = image.bottom; rect.top = image.top }
            else { rect.left = image.left; rect.right = image.right }
        } else if settings.vertical {
            rect.bottom = image.bottom + Int32(floor(Double(rect.bottom - image.bottom) / Double(n))) * n
            rect.top = image.bottom + Int32(ceil(Double(rect.top - image.bottom) / Double(n))) * n
        } else {
            rect.left = image.left + Int32(floor(Double(rect.left - image.left) / Double(n))) * n
            rect.right = image.left + Int32(ceil(Double(rect.right - image.left) / Double(n))) * n
        }
        rect.left = max(image.left, rect.left); rect.right = min(image.right, rect.right)
        rect.bottom = max(image.bottom, rect.bottom); rect.top = min(image.top, rect.top)
        sourceTileRect.pointee = rect
    }

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
        descriptor.storageMode = .private
        descriptor.usage = [.shaderRead, .shaderWrite]
        guard let intermediate = device.makeTexture(descriptor: descriptor),
              let command = gpu.queue.makeCommandBuffer() else {
            throw pixelSortError("Unable to allocate PixelSort rendering resources.")
        }
        let image = source.imagePixelBounds
        func relative(_ rect: FxRect) -> SIMD4<Int32> {
            SIMD4(rect.left - image.left, rect.bottom - image.bottom, rect.right - rect.left, rect.top - rect.bottom)
        }
        let dest = relative(bounds)
        let blockLength = max(1, settings.length)
        let first = (settings.vertical ? dest.y : dest.x) / blockLength
        let end = settings.vertical ? dest.y + dest.w : dest.x + dest.z
        let blocks = (end + blockLength - 1) / blockLength - first
        var uniforms = PixelSortUniforms()
        uniforms.sourceRect = relative(source.tilePixelBounds)
        uniforms.destinationRect = dest
        uniforms.configuration = SIMD4(settings.vertical ? 1 : 0, settings.reverse ? 1 : 0, settings.length, source.imageOrigin == kFxImageOrigin_TOP_LEFT ? 1 : 0)
        uniforms.dispatchInfo = SIMD4(destinationImage.imageOrigin == kFxImageOrigin_TOP_LEFT ? 1 : 0, first, settings.vertical ? dest.x : dest.y, 0)
        uniforms.controls = SIMD4(Float(settings.lower), Float(settings.upper), Float(settings.mix), 0)
        if settings.length == 0 {
            let axisLength = Int(settings.vertical ? image.top - image.bottom : image.right - image.left)
            try gpu.encodeFullSort(command: command, source: input, destination: intermediate,
                                   uniforms: uniforms, axisLength: axisLength)
        } else {
            guard let compute = command.makeComputeCommandEncoder() else {
                throw pixelSortError("Unable to create the block sort encoder.")
            }
            compute.setComputePipelineState(gpu.compute)
            compute.setTexture(input, index: 0)
            compute.setTexture(intermediate, index: 1)
            compute.setBytes(&uniforms, length: MemoryLayout<PixelSortUniforms>.stride, index: 0)
            compute.dispatchThreadgroups(MTLSize(width: Int(blocks), height: settings.vertical ? width : height, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(settings.length), height: 1, depth: 1))
            compute.endEncoding()
        }

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
        command.waitUntilCompleted()
        if command.status != .completed { throw command.error ?? pixelSortError("PixelSort GPU rendering failed.") }
    }
}
