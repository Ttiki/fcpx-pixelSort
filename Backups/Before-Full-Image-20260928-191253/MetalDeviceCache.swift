import Foundation

func pixelSortError(_ message: String) -> NSError {
    NSError(domain: "com.clementcombier.PixelSort", code: 1,
            userInfo: [NSLocalizedDescriptionKey: message])
}

final class PixelSortGPU {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let compute: MTLComputePipelineState
    let render: MTLRenderPipelineState

    init(device: MTLDevice, format: MTLPixelFormat) throws {
        self.device = device
        guard let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let sort = library.makeFunction(name: "pixelSort"),
              let vertex = library.makeFunction(name: "vertexShader"),
              let fragment = library.makeFunction(name: "fragmentShader") else {
            throw pixelSortError("Unable to load the PixelSort Metal shaders.")
        }
        self.queue = queue
        compute = try device.makeComputePipelineState(function: sort)
        guard compute.maxTotalThreadsPerThreadgroup >= 256 else {
            throw pixelSortError("This GPU does not support the required sorting block size.")
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        descriptor.colorAttachments[0].pixelFormat = format
        render = try device.makeRenderPipelineState(descriptor: descriptor)
    }
}

final class MetalDeviceCache {
    static let deviceCache = MetalDeviceCache()
    private let lock = NSLock()
    private var entries: [String: PixelSortGPU] = [:]

    func gpu(registryID: UInt64, format: MTLPixelFormat) throws -> PixelSortGPU {
        lock.lock()
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
