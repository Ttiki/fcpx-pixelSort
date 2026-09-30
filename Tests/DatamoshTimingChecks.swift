// Exercises the actual scheduler helper with rational frame rates, clip starts, and bypass.
import Foundation
import CoreMedia

@main struct DatamoshTimingChecks {
    static func main() throws {
        var cases = 0
        for duration in [CMTime(value:1,timescale:24),CMTime(value:1,timescale:25),CMTime(value:1,timescale:30),CMTime(value:1001,timescale:30000)] {
            for start in [CMTime.zero,CMTime(value:17,timescale:3),CMTime(value:-11,timescale:2)] {
                for frame: Int32 in [0,1,4,120] {
                    let render=CMTimeAdd(start,CMTimeMultiply(duration,multiplier:frame))
                    let result=try DatamoshTiming.past(render:render,start:start,duration:duration,frames:6)
                    precondition(CMTimeCompare(result.previous,CMTimeAdd(start,CMTimeMultiply(duration,multiplier:max(0,frame-1)))) == 0)
                    precondition(CMTimeCompare(result.history,CMTimeAdd(start,CMTimeMultiply(duration,multiplier:max(0,frame-6)))) == 0)
                    let requests=DatamoshTiming.requests(current:render,previous:result.previous,history:result.history,enabled:true)
                    precondition(requests.count == (frame == 0 ? 1 : (frame == 1 ? 2 : 3)))
                    precondition(requests.allSatisfy { CMTimeCompare($0,render) <= 0 && CMTimeCompare($0,start) >= 0 })
                    precondition(DatamoshTiming.requests(current:render,previous:result.previous,history:result.history,enabled:false).count == 1)
                    cases += 1
                }
            }
        }
        let early=try DatamoshTiming.past(render:.zero,start:CMTime(value:3,timescale:1),duration:CMTime(value:1,timescale:30),frames:6)
        precondition(CMTimeCompare(early.history,.zero) == 0)
        do { _ = try DatamoshTiming.past(render:.zero,start:.zero,duration:.invalid,frames:6);fatalError("Invalid timing must fail") } catch {}
        print("PASS: \(cases) scheduler cases, plus pre-start passthrough and invalid-time rejection.")
    }
}
