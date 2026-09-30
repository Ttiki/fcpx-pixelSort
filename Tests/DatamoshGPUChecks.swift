// Compares the production two-pass temporal renderer with a scalar CPU reference.
// Synthetic translated frames provide known motion; independent delayed content exposes history use.
import Foundation
import Metal
import simd

@main struct DatamoshGPUChecks {
    /// Executes deterministic comparisons over bypass, mix, block size, seeds, tiles, and origins.
    static func main() throws {
        guard CommandLine.arguments.count == 2 else { fatalError("Pass the built default.metallib path.") }
        let device=MTLCreateSystemDefaultDevice()!
        let library=try device.makeLibrary(URL:URL(fileURLWithPath:CommandLine.arguments[1]))
        let gpu=try PixelSortGPU(device:device,format:.rgba32Float,library:library)
        let w=73,h=53
        /// Matches the specified wrapping integer seed mapping; motion/reference logic stays scalar.
        func hash(_ value: UInt32) -> UInt32 {
            var x=value; x ^= x >> 16; x=x &* 0x7feb352d; x ^= x >> 15; x=x &* 0x846ca68b; return x ^ (x >> 16)
        }
        /// Transparent border samples represent image content beyond the decoded frame.
        func read(_ pixels: [SIMD4<Float>], _ x: Int, _ y: Int) -> SIMD4<Float> { x >= 0 && x < w && y >= 0 && y < h ? pixels[y*w+x] : .zero }
        let previous: [SIMD4<Float>] = (0..<(w*h)).map { i in
            let k=hash(UInt32(i)),a: Float = i%11 == 0 ? 0 : (i%7 == 0 ? 0.5 : 1)
            return SIMD4(Float(k&15)/8*a,Float((k>>4)&15)/8*a,Float((k>>8)&15)/8*a,a)
        }
        let current=(0..<(w*h)).map { read(previous,$0%w-4,$0/w-4) }
        let history: [SIMD4<Float>] = (0..<(w*h)).map { i in
            let a: Float = i%5 == 0 ? 0 : 1
            return SIMD4(Float(i%7)/4*a,Float(i%13)/8*a,Float(i%3)/2*a,a)
        }
        /// Computes the expected per-block displacement with no GPU buffers or encoding involved.
        func reference(_ u: DatamoshUniforms, history old: [SIMD4<Float>]) -> [SIMD4<Float>] {
            let block=Int(u.imageInfo.z),step=Int(u.imageInfo.w),cols=(w+block-1)/block,rows=(h+block-1)/block
            var vectors=[SIMD2<Int>](repeating:.zero,count:cols*rows)
            if u.controls.x > 0 && u.controls.y > 0 && u.controls.z > 0 {
                for by in 0..<rows { for bx in 0..<cols {
                    /// Samples nine positions, clamping partial edge blocks to the actual image.
                    func cost(_ dx: Int,_ dy: Int) -> Float {
                        var result: Float=0
                        for y in 1...3 { for x in 1...3 {
                            let px=min(w-1,bx*block+x*block/4),py=min(h-1,by*block+y*block/4)
                            let difference=abs(read(current,px,py)-read(previous,px+dx,py+dy))
                            result += simd_dot(difference,SIMD4<Float>(0.2126,0.7152,0.0722,0.25))
                        }}
                        return result
                    }
                    var best=SIMD2<Int>.zero,bestCost=cost(0,0)
                    for y in -4...4 { for x in -4...4 {
                        let dx=x*step,dy=y*step,c=cost(dx,dy)
                        if c < bestCost || (c == bestCost && dx*dx+dy*dy < best.x*best.x+best.y*best.y) { best=SIMD2(dx,dy);bestCost=c }
                    }}
                    vectors[by*cols+bx]=best
                }}
            }
            return (0..<(w*h)).map { i in
                let original=current[i],x=i%w,y=i/w,bx=x/block,by=y/block
                if u.controls.x <= 0 || u.controls.z <= 0 { return original }
                let k=hash(UInt32(bx) ^ hash(UInt32(by)) ^ hash(u.random.x))
                guard Float(k&0xffffff)/16777216 < u.controls.x else { return original }
                let v=vectors[by*cols+bx]
                let dx=Int((Float(v.x)*u.controls.y*8).rounded()),dy=Int((Float(v.y)*u.controls.y*8).rounded())
                let result=read(old,x+dx,y+dy)
                return u.controls.z >= 1 ? result : original+(result-original)*u.controls.z
            }
        }
        /// Uploads three independently oriented source tiles and reads back canonical output rows.
        func render(_ uniform: DatamoshUniforms, cropped: Bool, old: [SIMD4<Float>]) throws -> [SIMD4<Float>] {
            var u=uniform
            let dx=cropped ? 11 : 0,dy=cropped ? 9 : 0,dw=cropped ? w-22 : w,dh=cropped ? h-18 : h
            let block=Int(u.imageInfo.z),search=Int(u.imageInfo.w)*4
            let halo=block+search+Int(ceil(Float(search)*u.controls.y*8))+1
            let sx=max(0,dx-halo),sy=max(0,dy-halo),sw=min(w,dx+dw+halo)-sx,sh=min(h,dy+dh+halo)-sy
            let r=SIMD4<Int32>(Int32(sx),Int32(sy),Int32(sw),Int32(sh))
            u.currentRect=r;u.previousRect=r;u.historyRect=r
            u.destinationRect=SIMD4(Int32(dx),Int32(dy),Int32(dw),Int32(dh))
            u.grid=SIMD4(Int32(dx/block),Int32(dy/block),Int32((dx+dw+block-1)/block-dx/block),Int32((dy+dh+block-1)/block-dy/block))
            /// Source textures expose shader-read access only, like host-provided image surfaces.
            func texture(_ pixels: [SIMD4<Float>], _ top: Int32) -> MTLTexture {
                let d=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rgba32Float,width:sw,height:sh,mipmapped:false)
                d.storageMode = .shared;d.usage = [.shaderRead]
                let t=device.makeTexture(descriptor:d)!
                var data=[SIMD4<Float>](repeating:.zero,count:sw*sh)
                for y in 0..<sh { for x in 0..<sw { data[(top != 0 ? sh-1-y : y)*sw+x]=pixels[(y+sy)*w+x+sx] } }
                data.withUnsafeBytes { t.replace(region:MTLRegionMake2D(0,0,sw,sh),mipmapLevel:0,withBytes:$0.baseAddress!,bytesPerRow:sw*16) }
                return t
            }
            let a=texture(current,u.origins.x),b=texture(previous,u.origins.y),c=texture(old,u.origins.z)
            let d=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rgba32Float,width:dw,height:dh,mipmapped:false)
            d.storageMode = .shared;d.usage = [.shaderWrite,.shaderRead]
            let output=device.makeTexture(descriptor:d)!,command=gpu.queue.makeCommandBuffer()!
            try gpu.encodeDatamosh(command:command,current:a,previous:b,history:c,destination:output,uniforms:u)
            command.commit();command.waitUntilCompleted()
            precondition(command.status == .completed,"Datamosh GPU failed: \(String(describing:command.error))")
            var raw=[SIMD4<Float>](repeating:.zero,count:dw*dh)
            raw.withUnsafeMutableBytes { output.getBytes($0.baseAddress!,bytesPerRow:dw*16,from:MTLRegionMake2D(0,0,dw,dh),mipmapLevel:0) }
            var result=raw
            for y in 0..<dh { for x in 0..<dw { result[y*dw+x]=raw[(u.origins.w != 0 ? dh-1-y : y)*dw+x] } }
            return result
        }
        var cases=0
        for amount: Float in [0,0.65,1] { for mix: Float in [0,0.4,1] { for motion: Float in [0,0.75] { for block: Int32 in [8,16] { for seed: UInt32 in [1,98] { for origin: Int32 in [0,1] { for cropped in [false,true] {
            var u=DatamoshUniforms();u.imageInfo=SIMD4(Int32(w),Int32(h),block,max(1,block/8));u.origins=SIMD4(origin,1-origin,origin,1-origin)
            u.controls=SIMD4(amount,motion,mix,0);u.random=SIMD4(seed,0,0,0)
            let expected=reference(u,history:history),actual=try render(u,cropped:cropped,old:history)
            let dx=cropped ? 11 : 0,dy=cropped ? 9 : 0,dw=cropped ? w-22 : w,dh=cropped ? h-18 : h
            for y in 0..<dh { for x in 0..<dw {
                precondition(simd_reduce_max(abs(actual[y*dw+x]-expected[(y+dy)*w+x+dx])) < 0.00002,"Mismatch case \(cases), position \(x),\(y)")
            }}
            cases += 1
        }}}}}}}
        var u=DatamoshUniforms();u.imageInfo=SIMD4(Int32(w),Int32(h),8,1);u.controls=SIMD4(1,0,1,0);u.random=SIMD4(1,0,0,0)
        let frozen=try render(u,cropped:false,old:history)
        precondition(frozen == history,"Motion zero and Amount/Mix one must use delayed pixels exactly")
        let repeated=try render(u,cropped:false,old:history)
        precondition(frozen == repeated,"Output must not depend on render order")
        let replaced=try render(u,cropped:false,old:current)
        precondition(replaced == current && replaced != frozen,"Changing only the history frame must change the temporal result")
        print("PASS: \(cases) Datamosh GPU/reference cases plus delayed-frame replacement and deterministic replay.")
    }
}
