import Foundation
import Testing
@testable import MLXAudioSTT

/// Consumer contracts that must survive upstream synchronization. No model download.
struct VoxtStreamingContractTests {
    @Test func endedPreservesStructuredOutput() async {
        let segment = STTTranscriptSegment(
            text: "hello", startTime: 1.25, endTime: 2.5,
            speakerID: "speaker-1", language: "en", confidence: 0.9
        )
        let output = STTOutput(
            text: "hello", segments: [segment], language: "en",
            languageProvenance: .detected, generationTokens: 2
        )
        let (stream, continuation) = AsyncStream<TranscriptionEvent>.makeStream()
        continuation.yield(.ended(output))
        continuation.finish()
        var endedCount = 0
        for await event in stream {
            guard case .ended(let result) = event else {
                Issue.record("Expected the structured terminal event")
                continue
            }
            endedCount += 1
            #expect(result.text == output.text)
            #expect(result.segments == [segment])
            #expect(result.language == "en")
            #expect(result.languageProvenance == .detected)
            #expect(result.generationTokens == 2)
        }
        #expect(endedCount == 1)
    }

    @Test func automaticLanguageAndConservativeKVSettingsRemainAvailable() {
        let live = StreamingConfig(language: nil, kvBits: 8, kvGroupSize: 64, quantizedKVStart: 256)
        let batch = STTGenerateParameters(language: nil, kvBits: 8, kvGroupSize: 64, quantizedKVStart: 256)
        #expect(live.language == nil)
        #expect(batch.language == nil)
        #expect(live.kvBits == batch.kvBits)
        #expect(live.kvGroupSize == batch.kvGroupSize)
        #expect(live.quantizedKVStart == batch.quantizedKVStart)
    }

    @Test func inferenceFailureIsNotAnEmptySuccessfulTranscript() {
        let failure = StreamingFailure(message: "decoder failed")
        let event = TranscriptionEvent.failed(failure)
        guard case .failed(let result) = event else {
            Issue.record("Failures must not be converted to ended(empty text)")
            return
        }
        #expect(result == failure)
    }
}
