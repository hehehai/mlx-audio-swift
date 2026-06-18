import Foundation
import MLX

/// Native live session for Nemotron ASR.
///
/// The session accepts incremental 16 kHz mono samples, keeps Nemotron's cache-aware
/// encoder state across feeds, and emits `TranscriptionEvent` updates that match the
/// existing local ASR live-session API.
public final class NemotronASRStreamingSession: @unchecked Sendable {
    private let model: NemotronASRModel
    private let config: StreamingConfig
    private let queue = DispatchQueue(label: "com.mlx-audio.nemotron-asr-streaming-session")
    private let startedAt = Date()

    public let events: AsyncStream<TranscriptionEvent>
    private let continuation: AsyncStream<TranscriptionEvent>.Continuation

    private var audioSamples: [Float] = []
    private var processedMelFrames = 0
    private var consumedMelFrames = 0
    private var emittedEncoderFrames = 0
    private var melCache: MLXArray?
    private var attnCache: [MLXArray?]
    private var convCache: [MLXArray?]
    private var results: [NemoAlignedToken] = []
    private var lastToken: Int
    private var decoderState: NemoLSTMState?
    private var globalEncoderTime = 0
    private var previousText = ""
    private var lastDecodeAt = Date.distantPast
    private var stopped = false
    private var cancelled = false

    public init(model: NemotronASRModel, config: StreamingConfig = StreamingConfig()) {
        self.model = model
        self.config = config
        self.attnCache = [MLXArray?](repeating: nil, count: model.encoder.layers.count)
        self.convCache = [MLXArray?](repeating: nil, count: model.encoder.layers.count)
        self.lastToken = model.blankTokenID

        let stream = AsyncStream<TranscriptionEvent>.makeStream()
        self.events = stream.stream
        self.continuation = stream.continuation
    }

    public func feedAudio(samples: [Float]) {
        guard !samples.isEmpty else { return }
        queue.async { [weak self] in
            guard let self, !self.stopped, !self.cancelled else { return }
            self.audioSamples.append(contentsOf: samples)

            let now = Date()
            guard now.timeIntervalSince(self.lastDecodeAt) >= self.config.decodeIntervalSeconds else {
                return
            }
            self.lastDecodeAt = now
            self.processAvailableAudio(final: false)
        }
    }

    public func stop() {
        queue.async { [weak self] in
            guard let self, !self.stopped, !self.cancelled else { return }
            self.stopped = true
            self.processAvailableAudio(final: true)
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

    private func processAvailableAudio(final: Bool) {
        guard !audioSamples.isEmpty else { return }

        let audio = MLXArray(audioSamples).asType(.float32)
        let mel = NemotronASRAudio.logMelSpectrogram(audio, config: model.preprocessConfig)
        let totalMelFrames = mel.shape[1]
        guard processedMelFrames < totalMelFrames else { return }

        let newFrames = mel[0..., processedMelFrames..<totalMelFrames, 0...]
        processMelFrames(newFrames, final: final)
        processedMelFrames = totalMelFrames
    }

    private func processMelFrames(_ melFrames: MLXArray, final: Bool) {
        var features = melFrames
        if features.ndim == 2 {
            features = features.expandedDimensions(axis: 0)
        }
        features = features.asType(model.computeDType)

        let subsamplingFactor = model.encoderConfig.subsamplingFactor
        let rightContext = model.defaultAttContextSize.count > 1 ? model.defaultAttContextSize[1] : 13
        let chunkFrames = max(1, rightContext + 1)
        let chunkMelFrames = chunkFrames * subsamplingFactor
        let leftCache = model.defaultAttContextSize.first ?? 56
        let convLeft = model.encoderConfig.convKernelSize - 1
        let frameSeconds = Double(model.encoderConfig.subsamplingFactor * model.preprocessConfig.hopLength)
            / Double(model.preprocessConfig.sampleRate)
        let newFrameCount = features.shape[1]

        var cursor = 0
        while cursor < newFrameCount {
            let end = min(cursor + chunkMelFrames, newFrameCount)
            let current = features[0..., cursor..<end, 0...]
            let cacheLen = melCache?.shape[1] ?? 0
            let window = melCache == nil ? current : MLX.concatenated([melCache!, current], axis: 1)
            let windowLen = window.shape[1]
            let lengths = MLXArray([Int32(windowLen)]).asType(.int32)
            let subsampled = model.encoder.preEncode(window, lengths: lengths).0
            let isFinalSlice = final && end >= newFrameCount
            let globalStart = consumedMelFrames + cursor
            let globalEnd = consumedMelFrames + end
            let base = (globalStart - cacheLen) / subsamplingFactor
            let lo = emittedEncoderFrames - base
            let hi = isFinalSlice ? subsampled.shape[1] : (globalEnd / subsamplingFactor - base)

            melCache = window[0..., max(0, windowLen - nemoPreEncodeMelCache)..<windowLen, 0...]
            emittedEncoderFrames = base + max(lo, hi)
            cursor = end

            guard hi > lo else { continue }

            var hidden = subsampled[0..., lo..<hi, 0...]
            for index in model.encoder.layers.indices {
                let result = model.nemoStreamBlock(
                    model.encoder.layers[index],
                    hidden,
                    attnCache: attnCache[index],
                    convCache: convCache[index],
                    leftCache: leftCache,
                    convLeft: convLeft
                )
                hidden = result.0
                attnCache[index] = result.1
                convCache[index] = result.2
            }

            let prompted = model.applyPrompt(hidden, language: config.language)
            decodePromptedChunk(prompted, frameSeconds: frameSeconds)
        }

        consumedMelFrames += newFrameCount
        publishDisplayUpdate()
    }

    private func decodePromptedChunk(_ prompted: MLXArray, frameSeconds: Double) {
        let chunkLen = prompted.shape[1]
        var time = 0
        var newSymbols = 0

        while time < chunkLen {
            let frame = prompted[0..., time..<(time + 1), 0...]
            let currentToken: MLXArray? = lastToken == model.blankTokenID
                ? nil
                : MLXArray(Int32(lastToken)).reshaped([1, 1]).asType(.int32)
            let decoderOutput = model.decoder(currentToken, state: decoderState)
            let pred = decoderOutput.0.asType(frame.dtype)
            let proposedState: NemoLSTMState = (
                hidden: decoderOutput.1.hidden?.asType(frame.dtype),
                cell: decoderOutput.1.cell?.asType(frame.dtype)
            )
            let jointOutput = model.joint(frame, pred)
            let token = jointOutput.argMax(axis: -1).item(Int.self)
            let step = NemoDecodingLogic.rnntStep(
                predictedToken: token,
                blankToken: model.blankTokenID,
                time: time,
                newSymbols: newSymbols,
                maxSymbols: model.maxSymbols
            )

            if step.emittedToken {
                lastToken = token
                decoderState = proposedState
                if !NemotronASRTokenizer.isSpecialToken(token, vocabulary: model.vocabulary) {
                    results.append(
                        NemoAlignedToken(
                            id: token,
                            text: NemotronASRTokenizer.decode(tokens: [token], vocabulary: model.vocabulary),
                            start: Double(globalEncoderTime + time) * frameSeconds,
                            duration: frameSeconds
                        )
                    )
                }
            }

            time = step.nextTime
            newSymbols = step.nextNewSymbols
        }

        globalEncoderTime += chunkLen
    }

    private func publishDisplayUpdate() {
        let text = currentText()
        guard !text.isEmpty, text != previousText else { return }
        previousText = text
        continuation.yield(.displayUpdate(confirmedText: "", provisionalText: text))
        publishStats()
    }

    private func publishEnded() {
        let text = currentText()
        if !text.isEmpty {
            previousText = text
            continuation.yield(.ended(fullText: text))
        } else {
            continuation.yield(.ended(fullText: ""))
        }
    }

    private func currentText() -> String {
        NemoAlignment.sentencesToResult(NemoAlignment.tokensToSentences(results)).text
    }

    private func publishStats() {
        let elapsed = max(Date().timeIntervalSince(startedAt), 0.001)
        let audioSeconds = Double(audioSamples.count) / Double(model.preprocessConfig.sampleRate)
        continuation.yield(
            .stats(
                StreamingStats(
                    encodedWindowCount: max(globalEncoderTime, 0),
                    totalAudioSeconds: audioSeconds,
                    tokensPerSecond: Double(results.count) / elapsed,
                    realTimeFactor: elapsed / max(audioSeconds, 0.001),
                    peakMemoryGB: 0
                )
            )
        )
    }
}
