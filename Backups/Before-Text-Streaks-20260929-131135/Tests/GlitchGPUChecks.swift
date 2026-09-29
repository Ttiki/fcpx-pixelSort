import Foundation
import Metal
import AppKit
import simd

@main struct GlitchGPUChecks {
    static func main() throws {
        guard CommandLine.arguments.count >= 2 else { fatalError("Pass the built default.metallib path.") }
        let device = MTLCreateSystemDefaultDevice()!
        let library = try device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
        let gpu = try PixelSortGPU(device: device, format: .rgba32Float, library: library)
        func hash(_ value: UInt32) -> UInt32 {
            var x = value; x ^= x >> 16; x = x &* 0x7feb352d
            x ^= x >> 15; x = x &* 0x846ca68b; return x ^ (x >> 16)
        }
        func random(_ x: UInt32) -> Float { Float(hash(x) & 0x00ffffff) / 16777216 }
        func reference(_ pixels: [SIMD4<Float>], _ w: Int, _ h: Int, _ u: GlitchUniforms) -> [SIMD4<Float>] {
            func read(_ x: Int, _ y: Int) -> SIMD4<Float> { x >= 0 && x < w && y >= 0 && y < h ? pixels[y*w+x] : .zero }
            let base = u.configuration.y ^ hash(u.configuration.z &+ 0x9e3779b9)
            return (0..<(w*h)).map { i in
                let x = i % w, y = i / w, original = pixels[i]
                if u.controls.x <= 0 || u.controls.z <= 0 { return original }
                var result = original
                if u.configuration.x == 0 {
                    let size = max(1, Int((Float(h)*(0.005+0.12*u.controls.y)).rounded()))
                    let key = base ^ hash(UInt32(y/size)+17)
                    if random(key) < u.controls.w {
                        let displacement = (2*random(key ^ 0xa511e9b3)-1)*u.controls.x*Float(w)*0.25
                        result = read(x-Int(displacement.rounded()),y)
                    }
                } else {
                    let pulse: Float = u.configuration.w != 0 ? 0.25+0.75*random(base) : 1
                    let dx = Int((u.offset.x*pulse).rounded()), dy = Int((u.offset.y*pulse).rounded())
                    let red = read(x-dx,y-dy), blue = read(x+dx,y+dy)
                    result = SIMD4(red.x,original.y,blue.z,max(red.w,max(original.w,blue.w)))
                }
                return u.controls.z >= 1 ? result : original+(result-original)*u.controls.z
            }
        }
        func render(_ pixels: [SIMD4<Float>], _ w: Int, _ h: Int, _ uniform: GlitchUniforms, cropped: Bool = false) throws -> [SIMD4<Float>] {
            var u = uniform
            let dx = cropped ? 17 : 0, dy = cropped ? 9 : 0
            let dw = cropped ? w-34 : w, dh = cropped ? h-18 : h
            let haloX = Int(ceil(Float(w)*u.controls.x*(u.configuration.x == 0 ? 0.25 : 0.05)))+1
            let haloY = u.configuration.x == 0 ? 0 : haloX
            let sx = max(0,dx-haloX), sy = max(0,dy-haloY)
            let sw = min(w,dx+dw+haloX)-sx, sh = min(h,dy+dh+haloY)-sy
            u.sourceRect = SIMD4(Int32(sx),Int32(sy),Int32(sw),Int32(sh))
            u.destinationRect = SIMD4(Int32(dx),Int32(dy),Int32(dw),Int32(dh))
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float,width: sw,height: sh,mipmapped: false)
            descriptor.storageMode = .shared; descriptor.usage = [.shaderRead]
            let input = device.makeTexture(descriptor: descriptor)!
            var source = [SIMD4<Float>](repeating: .zero,count: sw*sh)
            for y in 0..<sh { for x in 0..<sw { source[(u.imageInfo.z != 0 ? sh-1-y : y)*sw+x] = pixels[(y+sy)*w+x+sx] } }
            source.withUnsafeBytes { input.replace(region: MTLRegionMake2D(0,0,sw,sh),mipmapLevel: 0,withBytes: $0.baseAddress!,bytesPerRow: sw*16) }
            descriptor.width = dw; descriptor.height = dh; descriptor.usage = [.shaderWrite,.shaderRead]
            let output = device.makeTexture(descriptor: descriptor)!
            let command = gpu.queue.makeCommandBuffer()!
            try gpu.encodeGlitch(command: command,source: input,destination: output,uniforms: u)
            command.commit(); command.waitUntilCompleted()
            guard command.status == .completed else { fatalError("GPU error: \(String(describing:command.error))") }
            var raw = [SIMD4<Float>](repeating: .zero,count: dw*dh)
            raw.withUnsafeMutableBytes { output.getBytes($0.baseAddress!,bytesPerRow: dw*16,from: MTLRegionMake2D(0,0,dw,dh),mipmapLevel: 0) }
            var result = [SIMD4<Float>](repeating: .zero,count: dw*dh)
            for y in 0..<dh { for x in 0..<dw { result[y*dw+x] = raw[(u.imageInfo.w != 0 ? dh-1-y : y)*dw+x] } }
            return result
        }
        let w = 97, h = 61
        let pixels: [SIMD4<Float>] = (0..<(w*h)).map { i in
            let alpha: Float = i % 7 == 0 ? 0 : (i % 3 == 0 ? 0.5 : 1)
            return SIMD4(Float(i%11)/8*alpha,Float(i%13)/8*alpha,Float(i%17)/8*alpha,alpha)
        }
        var cases = 0
        for effect: UInt32 in [0,1] { for amount: Float in [0,0.3,1] { for mix: Float in [0,0.4,1] { for tick: UInt32 in [0,17] { for seed: UInt32 in [1,92] { for origin: Int32 in [0,1] { for cropped in [false,true] {
            var u = GlitchUniforms()
            u.imageInfo = SIMD4(Int32(w),Int32(h),origin,1-origin)
            u.configuration = SIMD4(effect,seed,tick,tick == 0 ? 0 : 1)
            u.controls = SIMD4(amount,0.27,mix,0.65)
            u.offset = SIMD4(amount*Float(w)*0.04,amount*Float(w)*0.03,0,0)
            let expected = reference(pixels,w,h,u), actual = try render(pixels,w,h,u,cropped: cropped)
            let dx = cropped ? 17 : 0, dy = cropped ? 9 : 0, dw = cropped ? w-34 : w, dh = cropped ? h-18 : h
            for y in 0..<dh { for x in 0..<dw {
                precondition(simd_reduce_max(abs(actual[y*dw+x]-expected[(y+dy)*w+x+dx])) < 0.00001,"Mismatch case \(cases), \(x),\(y)")
            }}
            cases += 1
        }}}}}}}
        // Explicit white glyph on transparency: effects must move alpha, not just change brightness.
        let white: [SIMD4<Float>] = (0..<(w*h)).map { i in
            let x=i%w,y=i/w
            return (x >= 32 && x < 39 && y >= 8 && y < 53) || (x >= 32 && x < 65 && y >= 27 && y < 34) ? SIMD4(repeating: 1) : .zero
        }
        var u = GlitchUniforms(); u.imageInfo = SIMD4(Int32(w),Int32(h),0,0)
        u.configuration = SIMD4(0,1,0,0); u.controls = SIMD4(0.8,0.25,1,1)
        let tear = try render(white,w,h,u)
        precondition(tear != white,"White text must visibly tear")
        let repeatTear = try render(white,w,h,u)
        precondition(tear == repeatTear,"Repeated frame must be deterministic")
        u.configuration.z = 1
        let nextTear = try render(white,w,h,u)
        precondition(tear != nextTear,"Time must change the tear pattern")
        u.configuration = SIMD4(1,1,0,0); u.offset = SIMD4(4,0,0,0)
        let rgb = try render(white,w,h,u)
        precondition(rgb != white,"White text must split")
        precondition(zip(white,rgb).contains { pair in pair.0.w == 0 && pair.1.w > 0 && pair.1.x != pair.1.z },"Colored fringes must extend into transparent pixels")
        precondition(rgb.allSatisfy { $0.x <= $0.w && $0.y <= $0.w && $0.z <= $0.w },"SDR output must stay premultiplied")
        print("PASS: \(cases) GPU/reference cases plus transparent white-text, deterministic replay, and animation checks.")

        if CommandLine.arguments.count > 2 {
            let width=900,height=260
            let bitmap=NSBitmapImageRep(bitmapDataPlanes:nil,pixelsWide:width,pixelsHigh:height,bitsPerSample:8,samplesPerPixel:4,hasAlpha:true,isPlanar:false,colorSpaceName:.deviceRGB,bytesPerRow:width*4,bitsPerPixel:32)!
            NSGraphicsContext.saveGraphicsState(); NSGraphicsContext.current=NSGraphicsContext(bitmapImageRep:bitmap)
            NSColor.clear.setFill(); NSRect(x:0,y:0,width:width,height:height).fill(using:.copy)
            let text="SIGNAL LOST" as NSString
            text.draw(at:NSPoint(x:55,y:65),withAttributes:[.font:NSFont.monospacedSystemFont(ofSize:116,weight:.heavy),.foregroundColor:NSColor.white])
            NSGraphicsContext.restoreGraphicsState()
            let bytes=bitmap.bitmapData!
            let source: [SIMD4<Float>] = (0..<(width*height)).map { i in SIMD4(Float(bytes[i*4])/255,Float(bytes[i*4+1])/255,Float(bytes[i*4+2])/255,Float(bytes[i*4+3])/255) }
            var v=GlitchUniforms(); v.imageInfo=SIMD4(Int32(width),Int32(height),0,0);v.controls=SIMD4(0.45,0.1,1,0.8);v.configuration=SIMD4(0,5,0,0)
            let torn=try render(source,width,height,v)
            v.configuration.x=1;v.offset=SIMD4(12,0,0,0)
            let split=try render(source,width,height,v)
            let combined=try render(torn,width,height,v)
            let panels=[source,torn,split,combined]
            let labels=["ORIGINAL — WHITE TEXT ON TRANSPARENCY","SIGNAL TEAR","RGB SPLIT","SIGNAL TEAR + RGB SPLIT"]
            let canvas=NSBitmapImageRep(bitmapDataPlanes:nil,pixelsWide:width,pixelsHigh:(height+48)*4,bitsPerSample:8,samplesPerPixel:4,hasAlpha:true,isPlanar:false,colorSpaceName:.deviceRGB,bytesPerRow:width*4,bitsPerPixel:32)!
            NSGraphicsContext.saveGraphicsState(); NSGraphicsContext.current=NSGraphicsContext(bitmapImageRep:canvas)
            NSColor(calibratedWhite:0.045,alpha:1).setFill();NSRect(x:0,y:0,width:width,height:(height+48)*4).fill()
            for (index,panel) in panels.enumerated() {
                let image=NSBitmapImageRep(bitmapDataPlanes:nil,pixelsWide:width,pixelsHigh:height,bitsPerSample:8,samplesPerPixel:4,hasAlpha:true,isPlanar:false,colorSpaceName:.deviceRGB,bytesPerRow:width*4,bitsPerPixel:32)!
                for i in 0..<panel.count { for c in 0..<4 { image.bitmapData![i*4+c]=UInt8(max(0,min(255,(panel[i][c]*255).rounded()))) } }
                let y=(3-index)*(height+48)
                image.draw(in:NSRect(x:0,y:y,width:width,height:height))
                (labels[index] as NSString).draw(at:NSPoint(x:30,y:y+height+8),withAttributes:[.font:NSFont.monospacedSystemFont(ofSize:17,weight:.medium),.foregroundColor:NSColor(calibratedRed:0.3,green:0.9,blue:0.8,alpha:1)])
            }
            NSGraphicsContext.restoreGraphicsState()
            try canvas.representation(using:.png,properties:[:])!.write(to:URL(fileURLWithPath:CommandLine.arguments[2]))
            print("Preview saved: \(CommandLine.arguments[2])")
        }
    }
}
