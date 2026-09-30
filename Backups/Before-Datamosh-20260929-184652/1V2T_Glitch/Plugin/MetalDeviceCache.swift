// Shared GPU execution layer for the 1V2T collection. It caches compiled pipelines by
// device/output format and records compute passes without storing per-frame mutable state.
// Foundation provides locking and NSError; Metal provides devices, textures, queues, and encoders.

import Foundation
import Metal

/// Builds a localized error the host can display instead of crashing on a missing GPU resource.
func pixelSortError(_ message: String) -> NSError {
    NSError(domain: "com.clementcombier.PixelSort", code: 1,
            userInfo: [NSLocalizedDescriptionKey: message])
}

/// Owns reusable programs and a command queue for one device/output pixel format.
/// Every render gets its own command buffer, textures, uniforms, and scratch storage.
final class PixelSortGPU {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let compute: MTLComputePipelineState
    let render: MTLRenderPipelineState
    let initializeFull: MTLComputePipelineState
    let blocksFull: MTLComputePipelineState
    let mergeFull: MTLComputePipelineState
    let outputFull: MTLComputePipelineState
    let copyImage: MTLComputePipelineState
    let glitchEffect: MTLComputePipelineState
    let streakEffect: MTLComputePipelineState

    /// Loads the bundled shaders and compiles their compute/render pipelines once.
    /// Tests may inject the built Metal library to exercise exactly the production shader code.
    init(device: MTLDevice, format: MTLPixelFormat, library suppliedLibrary: MTLLibrary? = nil) throws {
        self.device = device
        guard let queue = device.makeCommandQueue(),
              let library = suppliedLibrary ?? device.makeDefaultLibrary(),
              let sort = library.makeFunction(name: "pixelSort"),
              let vertex = library.makeFunction(name: "vertexShader"),
              let fragment = library.makeFunction(name: "fragmentShader") else {
            throw pixelSortError("Unable to load the PixelSort Metal shaders.")
        }
        /// Resolves a named kernel and reports missing functions as a readable initialization error.
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw pixelSortError("Missing Metal function: \(name)")
            }
            return try device.makeComputePipelineState(function: function)
        }
        streakEffect = try pipeline("textStreaks")
        glitchEffect = try pipeline("glitchEffect")
        copyImage = try pipeline("pixelSortCopy")
        initializeFull = try pipeline("fullSortInitialize")
        blocksFull = try pipeline("fullSortBlocks")
        mergeFull = try pipeline("fullSortMerge")
        outputFull = try pipeline("fullSortOutput")
        self.queue = queue
        compute = try device.makeComputePipelineState(function: sort)
        guard compute.maxTotalThreadsPerThreadgroup >= 256, blocksFull.maxTotalThreadsPerThreadgroup >= 256 else {
            throw pixelSortError("This GPU does not support the required sorting block size.")
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        descriptor.colorAttachments[0].pixelFormat = format
        render = try device.makeRenderPipelineState(descriptor: descriptor)
    }

    /// Extends foreground pixels along each line with a seeded length, using one scan per line.
    /// A thread owns a complete row/column so gaps can inherit color from earlier pixels in O(n).
    func encodeTextStreaks(command: MTLCommandBuffer, source: MTLTexture, destination: MTLTexture,
                           uniforms: PixelSortUniforms, axisLength: Int, seed: UInt32) throws {
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw pixelSortError("Unable to create the text streak encoder.")
        }
        var u = uniforms
        var details = SIMD4<UInt32>(UInt32(axisLength), seed, 0, 0)
        let lines = Int(u.configuration.x != 0 ? u.destinationRect.z : u.destinationRect.w)
        encoder.label = "PixelSort text streaks"
        encoder.setComputePipelineState(streakEffect)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(destination, index: 1)
        encoder.setBytes(&u, length: MemoryLayout<PixelSortUniforms>.stride, index: 0)
        encoder.setBytes(&details, length: MemoryLayout.size(ofValue: details), index: 1)
        encoder.dispatchThreads(MTLSize(width: lines, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(64, streakEffect.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Schedules one GPU invocation per output pixel for Signal Tear or RGB Split.
    /// The uniform effect selector lets both filters share texture and dispatch plumbing.
    func encodeGlitch(command: MTLCommandBuffer, source: MTLTexture, destination: MTLTexture,
                      uniforms: GlitchUniforms) throws {
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw pixelSortError("Unable to create the glitch render encoder.")
        }
        var u = uniforms
        encoder.label = "1V2T Glitch"
        encoder.setComputePipelineState(glitchEffect)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(destination, index: 1)
        encoder.setBytes(&u, length: MemoryLayout<GlitchUniforms>.stride, index: 0)
        encoder.dispatchThreads(MTLSize(width: destination.width, height: destination.height, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(256, glitchEffect.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Schedules an exact pixel-copy path, translating tile offsets and texture orientations.
    /// This is used when a sorting operation would have no visible effect.
    func encodeCopy(command: MTLCommandBuffer, source: MTLTexture, destination: MTLTexture,
                    uniforms: PixelSortUniforms) throws {
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw pixelSortError("Unable to create the image copy encoder.")
        }
        var u = uniforms
        encoder.setComputePipelineState(copyImage)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(destination, index: 1)
        encoder.setBytes(&u, length: MemoryLayout<PixelSortUniforms>.stride, index: 0)
        encoder.dispatchThreads(MTLSize(width: destination.width, height: destination.height, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(256, copyImage.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Sorts complete lines while limiting scratch memory to 64 lines at a time.
    /// It initializes segment keys, sorts 256-entry groups, merges them, then gathers source colors.
    func encodeFullSort(command: MTLCommandBuffer, source: MTLTexture, destination: MTLTexture,
                        uniforms: PixelSortUniforms, axisLength: Int) throws {
        guard axisLength > 0 else { throw pixelSortError("Invalid full-line dimensions.") }
        let vertical = uniforms.configuration.x != 0
        let lines = Int(vertical ? uniforms.destinationRect.z : uniforms.destinationRect.w)
        let outputAxis = Int(vertical ? uniforms.destinationRect.w : uniforms.destinationRect.z)
        // Padding makes every local sorting group complete; sentinel keys sort after real pixels.
        let padded = ((axisLength + 255) / 256) * 256
        // Scratch storage is bounded by 64 lines rather than the entire video frame.
        let batchSize = min(lines, 64)
        let bytes = padded * batchSize * 16 // FullSortItem: four 32-bit fields.
        guard bytes <= device.maxBufferLength,
              let firstBuffer = device.makeBuffer(length: bytes, options: .storageModePrivate),
              let secondBuffer = device.makeBuffer(length: bytes, options: .storageModePrivate) else {
            throw pixelSortError("Not enough GPU memory for full-image sorting.")
        }
        /// Starts a labeled compute pass using a chosen pipeline; each pass separates dependent work.
        func encoder(_ pipeline: MTLComputePipelineState, _ label: String) throws -> MTLComputeCommandEncoder {
            guard let encoder = command.makeComputeCommandEncoder() else {
                throw pixelSortError("Unable to create a full-image sort encoder.")
            }
            encoder.label = label
            encoder.setComputePipelineState(pipeline)
            return encoder
        }
        for firstLine in stride(from: 0, to: lines, by: batchSize) {
            let count = min(batchSize, lines - firstLine)
            var u = uniforms
            u.dispatchInfo.z += Int32(firstLine)
            var info = SIMD4<UInt32>(UInt32(axisLength), UInt32(padded), 256, UInt32(count))
            let initialize = try encoder(initializeFull, "PixelSort full-line segments")
            initialize.setTexture(source, index: 0)
            initialize.setBuffer(firstBuffer, offset: 0, index: 0)
            initialize.setBytes(&u, length: MemoryLayout<PixelSortUniforms>.stride, index: 1)
            initialize.setBytes(&info, length: MemoryLayout.size(ofValue: info), index: 2)
            initialize.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(64, initializeFull.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            initialize.endEncoding()

            let local = try encoder(blocksFull, "PixelSort initial sorted runs")
            local.setBuffer(firstBuffer, offset: 0, index: 0)
            local.setBytes(&info, length: MemoryLayout.size(ofValue: info), index: 1)
            local.dispatchThreadgroups(MTLSize(width: padded / 256, height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            local.endEncoding()

            var input = firstBuffer, output = secondBuffer
            var runLength = 256
            while runLength < padded {
                info.z = UInt32(runLength)
                let merge = try encoder(mergeFull, "PixelSort merge \(runLength)-pixel runs")
                merge.setBuffer(input, offset: 0, index: 0)
                merge.setBuffer(output, offset: 0, index: 1)
                merge.setBytes(&info, length: MemoryLayout.size(ofValue: info), index: 2)
                merge.dispatchThreads(MTLSize(width: padded * count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, mergeFull.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                merge.endEncoding()
                // Ping-pong buffers keep reads separate from parallel writes during a merge.
                swap(&input, &output)
                runLength *= 2
            }
            let finish = try encoder(outputFull, "PixelSort full-line output")
            finish.setTexture(source, index: 0)
            finish.setTexture(destination, index: 1)
            finish.setBuffer(input, offset: 0, index: 0)
            finish.setBytes(&u, length: MemoryLayout<PixelSortUniforms>.stride, index: 1)
            finish.setBytes(&info, length: MemoryLayout.size(ofValue: info), index: 2)
            finish.dispatchThreads(MTLSize(width: outputAxis, height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, outputFull.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            finish.endEncoding()
        }
    }

}

/// Shares immutable GPU pipelines across filter instances, keyed by registry ID and output format.
/// A lock serializes cache creation because the host may render multiple frames concurrently.
final class MetalDeviceCache {
    static let deviceCache = MetalDeviceCache()
    private let lock = NSLock()
    private var entries: [String: PixelSortGPU] = [:]

    /// Returns an existing compatible pipeline set or creates it for the host-selected device.
    func gpu(registryID: UInt64, format: MTLPixelFormat) throws -> PixelSortGPU {
        lock.lock()
        // Release on both success and thrown initialization errors.
        defer { lock.unlock() }
        let key = "\(registryID):\(format.rawValue)"
        if let entry = entries[key] { return entry }
        guard let device = MTLCopyAllDevices().first(where: { $0.registryID == registryID }) else {
            throw pixelSortError("The host application's Metal device is unavailable.")
        }
        let entry = try PixelSortGPU(device: device, format: format)
        entries[key] = entry
        return entry
    }
}
