import Foundation

/// Native live session for Nemotron ASR.
///
/// The session accepts incremental 16 kHz mono samples, keeps Nemotron's cache-aware
/// encoder state across feeds, and emits `TranscriptionEvent` updates that match the
/// existing local ASR live-session API.
public final class NemotronASRStreamingSession: @unchecked Sendable {
    private let config: StreamingConfig
    private let session: NemotronASRStreamSession
    private let queue = DispatchQueue(label: "com.mlx-audio.nemotron-asr-streaming-session")
    private let startedAt = Date()

    public let events: AsyncStream<TranscriptionEvent>
    private let continuation: AsyncStream<TranscriptionEvent>.Continuation

    private var pendingAudioSamples: [Float] = []
    private var totalAudioSampleCount = 0
    private var previousText = ""
    private var lastDecodeAt = Date.distantPast
    private var stopped = false
    private var cancelled = false

    public init(model: NemotronASRModel, config: StreamingConfig = StreamingConfig()) {
        self.config = config
        self.session = model.makeStreamSession(
            language: config.language,
            chunkMs: Self.chunkMilliseconds(for: config.delayPreset)
        )

        let stream = AsyncStream<TranscriptionEvent>.makeStream()
        self.events = stream.stream
        self.continuation = stream.continuation
    }

    /// Maps the shared delay presets to Nemotron's supported streaming chunk ladder.
    public static func chunkMilliseconds(for preset: DelayPreset) -> Int {
        switch preset {
        case .realtime:
            return 160
        case .agent:
            return 560
        case .subtitle:
            return 1120
        case .custom(let milliseconds):
            let supported = [80, 160, 320, 560, 1120]
            return supported.min {
                abs(Double($0) - Double(milliseconds)) < abs(Double($1) - Double(milliseconds))
            } ?? 560
        }
    }

    public func feedAudio(samples: [Float]) {
        guard !samples.isEmpty else { return }
        queue.async { [weak self] in
            guard let self, !self.stopped, !self.cancelled else { return }
            self.pendingAudioSamples.append(contentsOf: samples)
            self.totalAudioSampleCount += samples.count

            let now = Date()
            let chunkInterval = Double(Self.chunkMilliseconds(for: self.config.delayPreset)) / 1000
            let decodeInterval = min(max(self.config.decodeIntervalSeconds, 0.01), chunkInterval)
            guard now.timeIntervalSince(self.lastDecodeAt) >= decodeInterval else {
                return
            }
            self.lastDecodeAt = now
            self.processPendingAudio()
        }
    }

    public func stop() {
        queue.async { [weak self] in
            guard let self, !self.stopped, !self.cancelled else { return }
            self.stopped = true
            self.processPendingAudio()
            self.publish(self.session.finish())
            self.publishEnded()
            self.continuation.finish()
        }
    }

    public func cancel() {
        queue.async { [weak self] in
            guard let self, !self.cancelled else { return }
            self.cancelled = true
            self.continuation.finish()
        }
    }

    private func processPendingAudio() {
        guard !pendingAudioSamples.isEmpty else { return }
        let samples = pendingAudioSamples
        pendingAudioSamples.removeAll(keepingCapacity: true)
        publish(session.step(samples))
    }

    private func publish(_ delta: NemotronASRStreamSession.Delta) {
        guard !delta.text.isEmpty else { return }
        let text = session.text
        guard !text.isEmpty, text != previousText else { return }
        previousText = text
        continuation.yield(.displayUpdate(confirmedText: "", provisionalText: text))
        publishStats()
    }

    private func publishEnded() {
        let text = session.text
        previousText = text
        continuation.yield(.ended(fullText: text))
    }

    private func publishStats() {
        let elapsed = max(Date().timeIntervalSince(startedAt), 0.001)
        let audioSeconds = Double(totalAudioSampleCount) / 16_000
        continuation.yield(
            .stats(
                StreamingStats(
                    encodedWindowCount: 0,
                    totalAudioSeconds: audioSeconds,
                    tokensPerSecond: Double(session.tokens.count) / elapsed,
                    realTimeFactor: elapsed / max(audioSeconds, 0.001),
                    peakMemoryGB: 0
                )
            )
        )
    }
}
