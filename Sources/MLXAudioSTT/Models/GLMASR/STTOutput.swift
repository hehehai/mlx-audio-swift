//
//  STTOutput.swift
//  MLXAudioSTT
//
// Created by Prince Canuma on 04/01/2026.
//

import Foundation

// MARK: - STT Generation Events

/// Events emitted during speech-to-text streaming generation.
public enum STTGeneration: Sendable {
    /// A generated text token during transcription
    case token(String)
    /// Generation statistics
    case info(STTGenerationInfo)
    /// Final transcription result
    case result(STTOutput)
}

/// Information about the STT generation process.
public struct STTGenerationInfo: Sendable {
    public let promptTokenCount: Int
    public let generationTokenCount: Int
    public let prefillTime: TimeInterval
    public let generateTime: TimeInterval
    public let tokensPerSecond: Double
    public let peakMemoryUsage: Double

    public init(
        promptTokenCount: Int,
        generationTokenCount: Int,
        prefillTime: TimeInterval,
        generateTime: TimeInterval,
        tokensPerSecond: Double,
        peakMemoryUsage: Double
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.prefillTime = prefillTime
        self.generateTime = generateTime
        self.tokensPerSecond = tokensPerSecond
        self.peakMemoryUsage = peakMemoryUsage
    }

    public var summary: String {
        """
        Prompt:     \(promptTokenCount) tokens, \(String(format: "%.2f", Double(promptTokenCount) / max(prefillTime, 0.001))) tokens/s, \(String(format: "%.3f", prefillTime))s
        Generation: \(generationTokenCount) tokens, \(String(format: "%.2f", tokensPerSecond)) tokens/s, \(String(format: "%.3f", generateTime))s
        Peak Memory Usage: \(peakMemoryUsage) GB
        """
    }
}

/// Errors that can occur during STT generation.
public enum STTError: Error, LocalizedError {
    case modelNotInitialized(String)
    case generationFailed(String)
    case invalidInput(String)
    case audioProcessingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotInitialized(let message):
            return "Model not initialized: \(message)"
        case .generationFailed(let message):
            return "Generation failed: \(message)"
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        case .audioProcessingFailed(let message):
            return "Audio processing failed: \(message)"
        }
    }
}

// MARK: - STT Output

/// Describes how the language attached to an STT result was resolved.
public enum STTLanguageProvenance: String, Codable, Hashable, Sendable {
    /// The model detected the language from the audio.
    case detected
    /// The caller explicitly requested the language.
    case requested
    /// The language describes translated output rather than the spoken input.
    case outputTarget
    /// The model used its own default language.
    case modelDefault
    /// The producer did not expose enough information to determine provenance.
    case unknown
}

/// A type-safe transcription segment shared by all STT models.
public struct STTTranscriptSegment: Codable, Hashable, Sendable {
    public let text: String
    public let startTime: TimeInterval?
    public let endTime: TimeInterval?
    public let speakerID: String?
    public let language: String?
    public let confidence: Double?
    public let emotion: String?
    public let event: String?

    public init(
        text: String,
        startTime: TimeInterval? = nil,
        endTime: TimeInterval? = nil,
        speakerID: String? = nil,
        language: String? = nil,
        confidence: Double? = nil,
        emotion: String? = nil,
        event: String? = nil
    ) {
        self.text = text
        self.startTime = startTime
        self.endTime = endTime
        self.speakerID = speakerID
        self.language = language
        self.confidence = confidence
        self.emotion = emotion
        self.event = event
    }

    public var hasTiming: Bool {
        guard let startTime, let endTime else { return false }
        return startTime.isFinite && endTime.isFinite && endTime >= startTime
    }
}

/// Output from speech-to-text transcription.
public struct STTOutput: Sendable {
    /// The transcribed text.
    public let text: String

    /// Transcription segments with timing information (optional).
    public let segments: [STTTranscriptSegment]?

    /// Detected language (optional).
    public let language: String?

    /// How `language` was selected.
    public let languageProvenance: STTLanguageProvenance

    /// Number of tokens in the prompt.
    public let promptTokens: Int

    /// Number of tokens generated.
    public let generationTokens: Int

    /// Total number of tokens processed.
    public let totalTokens: Int

    /// Prompt processing tokens per second.
    public let promptTps: Double

    /// Generation tokens per second.
    public let generationTps: Double

    /// Total processing time in seconds.
    public let totalTime: Double

    /// Peak memory usage in GB.
    public let peakMemoryUsage: Double

    public init(
        text: String,
        segments: [STTTranscriptSegment]? = nil,
        language: String? = nil,
        languageProvenance: STTLanguageProvenance = .unknown,
        promptTokens: Int = 0,
        generationTokens: Int = 0,
        totalTokens: Int = 0,
        promptTps: Double = 0.0,
        generationTps: Double = 0.0,
        totalTime: Double = 0.0,
        peakMemoryUsage: Double = 0.0
    ) {
        self.text = text
        self.segments = segments
        self.language = language
        self.languageProvenance = languageProvenance
        self.promptTokens = promptTokens
        self.generationTokens = generationTokens
        self.totalTokens = totalTokens
        self.promptTps = promptTps
        self.generationTps = generationTps
        self.totalTime = totalTime
        self.peakMemoryUsage = peakMemoryUsage
    }
}

extension STTOutput: CustomStringConvertible {
    public var description: String {
        var result = "STTOutput:\n"
        result += "  text: \(text)\n"
        if let language = language {
            result += "  language: \(language)\n"
        }
        result += "  prompt_tokens: \(promptTokens)\n"
        result += "  generation_tokens: \(generationTokens)\n"
        result += "  total_tokens: \(totalTokens)\n"
        result += "  prompt_tps: \(String(format: "%.2f", promptTps))\n"
        result += "  generation_tps: \(String(format: "%.2f", generationTps))\n"
        result += "  total_time: \(String(format: "%.2f", totalTime))s\n"
        result += "  peak_memory_usage: \(String(format: "%.2f", peakMemoryUsage)) GB"
        return result
    }
}
