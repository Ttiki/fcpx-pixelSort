// Rational-time scheduling shared by the FxPlug controller and standalone timing tests.
// Keeping this independent of host APIs makes clip boundaries and fractional frame rates testable.
import Foundation
import CoreMedia

struct DatamoshTiming {
    /// Computes past samples in the host's time coordinate system without crossing the clip start.
    /// Invalid timing produces a readable error rather than requesting an undefined media position.
    static func past(render: CMTime, start: CMTime, duration: CMTime, frames: Int32) throws -> (previous: CMTime, history: CMTime) {
        guard render.isNumeric, start.isNumeric, duration.isNumeric, CMTimeCompare(duration,.zero) > 0, frames >= 1 else {
            throw NSError(domain:"1V2T.Datamosh",code:1,userInfo:[NSLocalizedDescriptionKey:"Invalid temporal input timing."])
        }
        // A render before the reported clip start has no usable history; never request a future frame.
        let lower = CMTimeCompare(start,render) > 0 ? render : start
        func sample(_ n: Int32) -> CMTime {
            let requested = CMTimeSubtract(render,CMTimeMultiply(duration,multiplier:n))
            return CMTimeCompare(requested,lower) < 0 ? lower : requested
        }
        return (sample(1),sample(frames))
    }

    /// Deduplicates clamped requests while keeping the current frame first; bypass requests no history.
    static func requests(current: CMTime, previous: CMTime, history: CMTime, enabled: Bool) -> [CMTime] {
        var times = [current]
        if enabled {
            for time in [previous,history] where !times.contains(where:{ CMTimeCompare($0,time) == 0 }) { times.append(time) }
        }
        return times
    }
}
