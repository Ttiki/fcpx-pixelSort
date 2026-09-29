// Verifies text streaks against an independent CPU reference and optionally renders a preview.
// The reference searches upstream for each output pixel; the GPU uses a single streaming scan.
// Different implementations help catch mistakes in direction, run color selection, and tile offsets.
import Foundation
import Metal
import AppKit
import simd

@main struct TextStreakChecks {
    /// Loads the built shaders, runs reference comparisons, then optionally draws a white-text preview.
    static func main() throws {
        guard CommandLine.arguments.count >= 2 else { fatalError("Pass default.metallib and optionally a preview PNG path.") }
        let device = MTLCreateSystemDefaultDevice()!
        let library = try device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
        let gpu = try PixelSortGPU(device: device, format: .rgba32Float, library: library)
        /// Reproduces the seed mapping using overflow arithmetic, independent of scan implementation.
        func hash(_ value: UInt32) -> UInt32 {
            var x=value; x ^= x >> 16; x = x &* 0x7feb352d
            x ^= x >> 15; x = x &* 0x846ca68b; return x ^ (x >> 16)
        }
        /// Returns expected colors by searching backwards from each gap into the nearest foreground run.
        func reference(_ pixels: [SIMD4<Float>], _ w: Int, _ h: Int, _ u: PixelSortUniforms, _ seed: UInt32) -> [SIMD4<Float>] {
            let vertical=u.configuration.x != 0, backwards=(u.configuration.y != 0) != vertical
            let axisLength=vertical ? h : w, maximum=u.configuration.z == 0 ? axisLength : Int(u.configuration.z)
            let step=backwards ? -1 : 1
            /// Converts line coordinates to the reference image’s canonical row-major storage.
            func index(_ axis: Int, _ cross: Int) -> Int { vertical ? axis*w+cross : cross*w+axis }
            /// Ignores near-transparent edge noise and non-finite colors as streak sources.
            func foreground(_ c: SIMD4<Float>) -> Bool { c.w > 0.01 && c.x.isFinite && c.y.isFinite && c.z.isFinite && c.w.isFinite }
            return (0..<(w*h)).map { i in
                let original=pixels[i]
                if maximum <= 1 || u.controls.z <= 0 || foreground(original) { return original }
                let axis=vertical ? i/w : i%w, cross=vertical ? i%w : i/w
                let random=Float(hash(UInt32(cross) ^ hash(seed)) & 0x00ffffff)/16777216
                let span=1+Int(random*Float(maximum))
                var cursor=axis-step, distance=1
                while cursor >= 0 && cursor < axisLength && distance <= span {
                    if foreground(pixels[index(cursor,cross)]) { break }
                    cursor -= step; distance += 1
                }
                guard cursor >= 0 && cursor < axisLength && distance <= span else { return original }
                var carried=pixels[index(cursor,cross)]
                // Search the rest of this connected run for the strongest alpha. Strict comparison
                // keeps the nearest sample on ties, matching a forward scan's most recent equal sample.
                cursor -= step
                while cursor >= 0 && cursor < axisLength && foreground(pixels[index(cursor,cross)]) {
                    let candidate=pixels[index(cursor,cross)]
                    if candidate.w > carried.w { carried=candidate }
                    cursor -= step
                }
                let result=original+carried*(1-original.w)
                return u.controls.z >= 1 ? result : original+(result-original)*u.controls.z
            }
        }
        /// Executes the production GPU encoder with full-axis source input and optionally cropped output.
        /// Texture row flips differ between source and destination to exercise both host conventions.
        func render(_ pixels: [SIMD4<Float>], _ w: Int, _ h: Int, _ uniform: PixelSortUniforms, _ seed: UInt32, cropped: Bool = false) throws -> [SIMD4<Float>] {
            var u=uniform
            let vertical=u.configuration.x != 0
            let dx=cropped ? 13 : 0, dy=cropped ? 7 : 0
            let dw=cropped ? w-26 : w, dh=cropped ? h-14 : h
            let sx=vertical ? dx : 0, sy=vertical ? 0 : dy
            let sw=vertical ? dw : w, sh=vertical ? h : dh
            u.sourceRect=SIMD4(Int32(sx),Int32(sy),Int32(sw),Int32(sh))
            u.destinationRect=SIMD4(Int32(dx),Int32(dy),Int32(dw),Int32(dh))
            u.dispatchInfo.z=Int32(vertical ? dx : dy)
            let descriptor=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rgba32Float,width:sw,height:sh,mipmapped:false)
            descriptor.storageMode = .shared; descriptor.usage = [.shaderRead]
            let input=device.makeTexture(descriptor:descriptor)!
            var source=[SIMD4<Float>](repeating:.zero,count:sw*sh)
            for y in 0..<sh { for x in 0..<sw { source[(u.configuration.w != 0 ? sh-1-y : y)*sw+x]=pixels[(y+sy)*w+x+sx] } }
            source.withUnsafeBytes { input.replace(region:MTLRegionMake2D(0,0,sw,sh),mipmapLevel:0,withBytes:$0.baseAddress!,bytesPerRow:sw*16) }
            descriptor.width=dw;descriptor.height=dh;descriptor.usage=[.shaderWrite,.shaderRead]
            let output=device.makeTexture(descriptor:descriptor)!
            let command=gpu.queue.makeCommandBuffer()!
            if u.configuration.z == 1 || u.controls.z <= 0 {
                try gpu.encodeCopy(command:command,source:input,destination:output,uniforms:u)
            } else {
                try gpu.encodeTextStreaks(command:command,source:input,destination:output,uniforms:u,axisLength:vertical ? h : w,seed:seed)
            }
            command.commit();command.waitUntilCompleted()
            precondition(command.status == .completed,"GPU failure: \(String(describing:command.error))")
            var raw=[SIMD4<Float>](repeating:.zero,count:dw*dh)
            raw.withUnsafeMutableBytes { output.getBytes($0.baseAddress!,bytesPerRow:dw*16,from:MTLRegionMake2D(0,0,dw,dh),mipmapLevel:0) }
            var result=raw
            for y in 0..<dh { for x in 0..<dw { result[y*dw+x]=raw[(u.dispatchInfo.x != 0 ? dh-1-y : y)*dw+x] } }
            return result
        }
        let w=97,h=61
        // Colored runs include full and partial alpha so trails cannot accidentally bleach edges.
        let pixels: [SIMD4<Float>] = (0..<(w*h)).map { i in
            let x=i%w,y=i/w
            let shape=(x>=20 && x<29 && y>=8 && y<52) || (x>=20 && x<75 && y>=25 && y<32) || (x>=59 && x<66 && y>=10 && y<48)
            guard shape else { return .zero }
            let a: Float = x%7 == 0 ? 0.25 : 1
            return SIMD4(a,Float(y%5)/4*a,Float(x%3)/2*a,a)
        }
        var cases=0
        for vertical: Int32 in [0,1] { for reverse: Int32 in [0,1] { for length: Int32 in [1,2,17,0] { for mix: Float in [0,0.4,1] { for seed: UInt32 in [1,92] { for origin: Int32 in [0,1] { for cropped in [false,true] {
            var u=PixelSortUniforms();u.configuration=SIMD4(vertical,reverse,length,origin)
            u.dispatchInfo=SIMD4(1-origin,0,0,0);u.controls=SIMD4(0.15,0.85,mix,0)
            let expected=reference(pixels,w,h,u,seed),actual=try render(pixels,w,h,u,seed,cropped:cropped)
            let dx=cropped ? 13 : 0,dy=cropped ? 7 : 0,dw=cropped ? w-26 : w,dh=cropped ? h-14 : h
            for y in 0..<dh { for x in 0..<dw {
                precondition(simd_reduce_max(abs(actual[y*dw+x]-expected[(y+dy)*w+x+dx])) < 0.00001,"Text streak mismatch case \(cases), \(x),\(y)")
            }}
            cases += 1
        }}}}}}}
        let white=pixels.map { SIMD4<Float>(repeating:$0.w) }
        var u=PixelSortUniforms();u.configuration=SIMD4(0,0,0,0);u.controls=SIMD4(0.15,0.85,1,0)
        let first=try render(white,w,h,u,1),again=try render(white,w,h,u,1),other=try render(white,w,h,u,92)
        precondition(first == again && first != other,"Seeded streaks must repeat and respond to seed changes")
        precondition(zip(white,first).contains { pair in pair.0.w == 0 && pair.1.w > 0 },"White pixels must stretch into transparent space despite sorting brightness thresholds")
        precondition(zip(white,first).allSatisfy { pair in pair.0.w <= 0.01 || pair.0 == pair.1 },"Foreground text must remain intact")
        print("PASS: \(cases) streak/reference cases plus white text, foreground preservation, and seed determinism.")

        if CommandLine.arguments.count > 2 {
            let width=900,height=260
            let bitmap=NSBitmapImageRep(bitmapDataPlanes:nil,pixelsWide:width,pixelsHigh:height,bitsPerSample:8,samplesPerPixel:4,hasAlpha:true,isPlanar:false,colorSpaceName:.deviceRGB,bytesPerRow:width*4,bitsPerPixel:32)!
            NSGraphicsContext.saveGraphicsState();NSGraphicsContext.current=NSGraphicsContext(bitmapImageRep:bitmap)
            NSColor.clear.setFill();NSRect(x:0,y:0,width:width,height:height).fill(using:.copy)
            ("GLITCH" as NSString).draw(at:NSPoint(x:175,y:80),withAttributes:[.font:NSFont.monospacedSystemFont(ofSize:110,weight:.heavy),.foregroundColor:NSColor.white])
            NSGraphicsContext.restoreGraphicsState()
            let bytes=bitmap.bitmapData!
            // AppKit bitmap rows start at the top; the test renderer uses canonical bottom-up rows.
            let source: [SIMD4<Float>] = (0..<(width*height)).map { i in
                let j=((height-1-i/width)*width+i%width)*4
                return SIMD4(Float(bytes[j])/255,Float(bytes[j+1])/255,Float(bytes[j+2])/255,Float(bytes[j+3])/255)
            }
            var v=PixelSortUniforms();v.configuration=SIMD4(0,0,150,0);v.controls=SIMD4(0.15,0.85,1,0)
            let horizontal=try render(source,width,height,v,7)
            v.configuration=SIMD4(1,0,80,0)
            let vertical=try render(source,width,height,v,7)
            let panels=[source,horizontal,vertical],labels=["ORIGINAL — WHITE TEXT","TEXT STREAKS — HORIZONTAL","TEXT STREAKS — VERTICAL"]
            let canvas=NSBitmapImageRep(bitmapDataPlanes:nil,pixelsWide:width,pixelsHigh:(height+48)*3,bitsPerSample:8,samplesPerPixel:4,hasAlpha:true,isPlanar:false,colorSpaceName:.deviceRGB,bytesPerRow:width*4,bitsPerPixel:32)!
            NSGraphicsContext.saveGraphicsState();NSGraphicsContext.current=NSGraphicsContext(bitmapImageRep:canvas)
            NSColor(calibratedWhite:0.04,alpha:1).setFill();NSRect(x:0,y:0,width:width,height:(height+48)*3).fill()
            for (index,panel) in panels.enumerated() {
                let image=NSBitmapImageRep(bitmapDataPlanes:nil,pixelsWide:width,pixelsHigh:height,bitsPerSample:8,samplesPerPixel:4,hasAlpha:true,isPlanar:false,colorSpaceName:.deviceRGB,bytesPerRow:width*4,bitsPerPixel:32)!
                // Convert canonical rows back to AppKit only when writing the demonstration image.
                for i in 0..<panel.count {
                    let j=(height-1-i/width)*width+i%width
                    for c in 0..<4 { image.bitmapData![j*4+c]=UInt8(max(0,min(255,(panel[i][c]*255).rounded()))) }
                }
                let y=(2-index)*(height+48)
                image.draw(in:NSRect(x:0,y:y,width:width,height:height))
                (labels[index] as NSString).draw(at:NSPoint(x:30,y:y+height+8),withAttributes:[.font:NSFont.monospacedSystemFont(ofSize:17,weight:.medium),.foregroundColor:NSColor(calibratedRed:0.3,green:0.9,blue:0.8,alpha:1)])
            }
            NSGraphicsContext.restoreGraphicsState()
            try canvas.representation(using:.png,properties:[:])!.write(to:URL(fileURLWithPath:CommandLine.arguments[2]))
            print("Saved text-streak preview.")
        }
    }
}
