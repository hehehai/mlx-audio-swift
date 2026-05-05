<a href="https://trendshift.io/repositories/20684" target="_blank"><img src="https://trendshift.io/api/badge/repositories/20684" alt="Blaizzy%2Fmlx-audio-swift | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

# MLX Audio Swift

A modular Swift SDK for audio processing with MLX on Apple Silicon

![Platform](https://img.shields.io/badge/platform-macOS%2014%2B%20%7C%20iOS%2017%2B-lightgrey)
![Swift](https://img.shields.io/badge/Swift-5.9%2B-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

## Architecture

MLXAudio follows a modular design allowing you to import only what you need:

- **MLXAudioCore**: Base types, protocols, and utilities
- **MLXAudioCodecs**: Audio codec implementations (SNAC, Encodec, Vocos, Mimi, DACVAE)
- **MLXAudioTTS**: Text-to-Speech models (Qwen3-TTS, Fish Audio S2 Pro, Soprano, VyvoTTS, Orpheus, Marvis TTS, Pocket TTS, Chatterbox, Echo TTS, KittenTTS, Kokoro, MOSS-TTS-Nano)
- **MLXAudioSTT**: Speech-to-Text models (Qwen3-ASR, Qwen3-ForcedAligner, Voxtral Realtime, Cohere Transcribe, Parakeet, GLM-ASR, Granite Speech, SenseVoice, FireRed ASR 2)
- **MLXAudioVAD**: Voice Activity Detection & Speaker Diarization (Sortformer, SmartTurn)
- **MLXAudioLID**: Spoken language identification (MMS-LID-256, VoxLingua107 ECAPA-TDNN)
- **MLXAudioSTS**: Speech-to-Speech, separation, and enhancement models (LFM2.5-Audio, SAM-Audio, MossFormer2-SE, DeepFilterNet)
- **MLXAudioG2P**: Grapheme-to-phoneme utilities for multilingual TTS pipelines
- **MLXAudioUI**: SwiftUI components for audio interfaces

## Installation

Add MLXAudio to your project using Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/Blaizzy/mlx-audio-swift.git", branch: "main")
]

// Import only what you need
.product(name: "MLXAudioTTS", package: "mlx-audio-swift"),
.product(name: "MLXAudioCore", package: "mlx-audio-swift")
```

## Quick Start

### Text-to-Speech

```swift
import MLXAudioTTS
import MLXAudioCore

// Load a TTS model from HuggingFace
let model = try await SopranoModel.fromPretrained("mlx-community/Soprano-80M-bf16")

// Generate audio
let audio = try await model.generate(
    text: "Hello from MLX Audio Swift!",
    parameters: GenerateParameters(
        maxTokens: 200,
        temperature: 0.7,
        topP: 0.95
    )
)

// Save to file
try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
```

### Speech-to-Text

```swift
import MLXAudioSTT
import MLXAudioCore

// Load audio file
let (sampleRate, audioData) = try loadAudioArray(from: audioURL)

// Load STT model
let model = try await GLMASRModel.fromPretrained("mlx-community/GLM-ASR-Nano-2512-4bit")

// Transcribe
let output = model.generate(audio: audioData)
print(output.text)
```

### Speaker Diarization

```swift
import MLXAudioVAD
import MLXAudioCore

// Load audio file
let (sampleRate, audioData) = try loadAudioArray(from: audioURL)

// Load diarization model
let model = try await SortformerModel.fromPretrained(
    "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"
)

// Detect who is speaking when
let output = try await model.generate(audio: audioData, threshold: 0.5)
for segment in output.segments {
    print("Speaker \(segment.speaker): \(segment.start)s - \(segment.end)s")
}
```

### Streaming Generation

```swift
for try await event in model.generateStream(text: text, parameters: parameters) {
    switch event {
    case .token(let token):
        print("Generated token: \(token)")
    case .audio(let audio):
        print("Final audio shape: \(audio.shape)")
    case .info(let info):
        print(info.summary)
    }
}
```

## Supported Models

This list reflects the model families currently wired up in the codebase under `Sources/` plus the top-level model factories such as `TTS.loadModel(...)` and `STS.loadModel(...)`.
For the full checkpoint matrix of each family, see the model-specific README linked below.

### TTS Models

| Model family | Current support in Swift | Model README | Example / default repo |
|--------------|--------------------------|--------------|------------------------|
| Qwen3-TTS | Base, CustomVoice, and VoiceDesign checkpoints; 0.6B and 1.7B variants documented | [Qwen3-TTS README](Sources/MLXAudioTTS/Models/Qwen3TTS/README.md) | [mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit) |
| Fish Audio S2 Pro | bf16 and 8bit checkpoints; reference voice cloning supported | [Fish Audio S2 Pro README](Sources/MLXAudioTTS/Models/FishSpeech/README.md) | [mlx-community/fish-audio-s2-pro-8bit](https://huggingface.co/mlx-community/fish-audio-s2-pro-8bit) |
| Soprano | Compact autoregressive TTS | [Soprano README](Sources/MLXAudioTTS/Models/Soprano/README.md) | [mlx-community/Soprano-80M-bf16](https://huggingface.co/mlx-community/Soprano-80M-bf16) |
| VyvoTTS | Qwen3-based English TTS | [VyvoTTS README](Sources/MLXAudioTTS/Models/Qwen3/README.md) | [mlx-community/VyvoTTS-EN-Beta-4bit](https://huggingface.co/mlx-community/VyvoTTS-EN-Beta-4bit) |
| Orpheus / Llama TTS | Llama-based speech LLM with named voices | [Llama TTS README](Sources/MLXAudioTTS/Models/Llama/README.md) | [mlx-community/orpheus-3b-0.1-ft-bf16](https://huggingface.co/mlx-community/orpheus-3b-0.1-ft-bf16) |
| Marvis TTS | Multi-voice conversational TTS (EN/FR/DE) | [Marvis TTS README](Sources/MLXAudioTTS/Models/Marvis/README.md) | [Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit](https://huggingface.co/Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit) |
| Pocket TTS | Lightweight CPU-friendly TTS with multiple built-in voices | [Pocket TTS README](Sources/MLXAudioTTS/Models/PocketTTS/README.md) | [mlx-community/pocket-tts](https://huggingface.co/mlx-community/pocket-tts) |
| Chatterbox TTS | Regular and Turbo variants; multilingual regular model plus quantized Turbo checkpoints | [Chatterbox README](Sources/MLXAudioTTS/Models/Chatterbox/README.md) | [mlx-community/chatterbox-turbo-fp16](https://huggingface.co/mlx-community/chatterbox-turbo-fp16) |
| Echo TTS | Diffusion TTS with short-clip voice cloning | [Echo TTS README](Sources/MLXAudioTTS/Models/EchoTTS/README.md) | [mlx-community/echo-tts-base](https://huggingface.co/mlx-community/echo-tts-base) |
| KittenTTS | mini / micro / nano family; verified MLX ports include mini and micro | [KittenTTS README](Sources/MLXAudioTTS/Models/StyleTTS2/KittenTTS/README.md) | [mlx-community/kitten-tts-mini-0.8](https://huggingface.co/mlx-community/kitten-tts-mini-0.8) |
| Kokoro | 82M multilingual non-autoregressive TTS; code also recognizes `kokoro-v1-*` style repos | [Kokoro README](Sources/MLXAudioTTS/Models/StyleTTS2/Kokoro/README.md) | [mlx-community/Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16) |
| MOSS-TTS-Nano | `moss_tts_nano` family supported by the factory; uses separate MOSS audio tokenizer assets | [MossTTSNano source](Sources/MLXAudioTTS/Models/MossTTSNano) | [mlx-community/MOSS-TTS-Nano](https://huggingface.co/mlx-community/MOSS-TTS-Nano) |

### STT Models

| Model family | Current support in Swift | Model README | Example / default repo |
|--------------|--------------------------|--------------|------------------------|
| Qwen3-ASR | 0.6B and 1.7B checkpoints across bf16 / 8bit / 6bit / 4bit | [Qwen3-ASR README](Sources/MLXAudioSTT/Models/Qwen3ASR/README.md) | [mlx-community/Qwen3-ASR-0.6B-4bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-4bit) |
| Qwen3-ForcedAligner | 0.6B forced-alignment checkpoints across bf16 / 8bit / 6bit / 4bit | [Qwen3-ASR README](Sources/MLXAudioSTT/Models/Qwen3ASR/README.md) | [mlx-community/Qwen3-ForcedAligner-0.6B-4bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-4bit) |
| Voxtral Realtime | fp16, 6bit, and 4bit realtime checkpoints | [Voxtral README](Sources/MLXAudioSTT/Models/VoxtralRealtime/README.md) | [mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit) |
| Cohere Transcribe | Cohere Transcribe 03-2026 encoder-decoder ASR | [Cohere Transcribe README](Sources/MLXAudioSTT/Models/CohereTranscribe/README.md) | [beshkenadze/cohere-transcribe-03-2026-mlx-fp16](https://huggingface.co/beshkenadze/cohere-transcribe-03-2026-mlx-fp16) |
| Parakeet | TDT, CTC, RNNT, and TDT-CTC variants from 110M through 1.1B | [Parakeet README](Sources/MLXAudioSTT/Models/Parakeet/README.md) | [mlx-community/parakeet-tdt-0.6b-v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) |
| GLM-ASR | Whisper-style encoder plus GLM/LLaMA-style decoder | [GLM-ASR README](Sources/MLXAudioSTT/Models/GLMASR/README.md) | [mlx-community/GLM-ASR-Nano-2512-4bit](https://huggingface.co/mlx-community/GLM-ASR-Nano-2512-4bit) |
| Granite Speech | ASR plus speech translation | [Granite Speech README](Sources/MLXAudioSTT/Models/GraniteSpeech/README.md) | [mlx-community/granite-4.0-1b-speech-5bit](https://huggingface.co/mlx-community/granite-4.0-1b-speech-5bit) |
| SenseVoice | ASR plus spoken language ID, emotion recognition, and audio event detection metadata | [SenseVoice README](Sources/MLXAudioSTT/Models/SenseVoice/README.md) | [mlx-community/SenseVoiceSmall](https://huggingface.co/mlx-community/SenseVoiceSmall) |
| FireRed ASR 2 | AED-style encoder-decoder ASR | [FireRed ASR 2 README](Sources/MLXAudioSTT/Models/FireRedASR2/README.md) | [mlx-community/FireRedASR2-AED-mlx](https://huggingface.co/mlx-community/FireRedASR2-AED-mlx) |

### STS / Enhancement / Separation Models

| Model family | Current support in Swift | Model README | Example / default repo |
|--------------|--------------------------|--------------|------------------------|
| LFM2.5-Audio | Multimodal audio model supporting TTS, STT, STS, and interleaved text/audio generation | [LFM Audio README](Sources/MLXAudioSTS/Models/LFMAudio/README.md) | [mlx-community/LFM2.5-Audio-1.5B-6bit](https://huggingface.co/mlx-community/LFM2.5-Audio-1.5B-6bit) |
| SAM-Audio | Text-guided source separation with short, long, and streaming APIs | [SAM Audio README](Sources/MLXAudioSTS/Models/SAMAudio/README.md) | [mlx-community/sam-audio-large-fp16](https://huggingface.co/mlx-community/sam-audio-large-fp16) |
| MossFormer2-SE | Speech enhancement / denoising | — | [starkdmi/MossFormer2-SE-fp16](https://huggingface.co/starkdmi/MossFormer2-SE-fp16) |
| DeepFilterNet | DeepFilterNet v1 / v2 / v3, offline and streaming enhancement | [DeepFilterNet README](Sources/MLXAudioSTS/Models/DeepFilterNet/README.md) | [mlx-community/DeepFilterNet-mlx](https://huggingface.co/mlx-community/DeepFilterNet-mlx) |

### LID Models

| Model family | Current support in Swift | Model README | Example / default repo |
|--------------|--------------------------|--------------|------------------------|
| MMS-LID-256 | Wav2Vec2-based spoken language identification across 256 languages | [MLXAudioLID README](Sources/MLXAudioLID/README.md) | [facebook/mms-lid-256](https://huggingface.co/facebook/mms-lid-256) |
| VoxLingua107 ECAPA-TDNN | Compact ECAPA-TDNN language ID model across 107 languages | [MLXAudioLID README](Sources/MLXAudioLID/README.md) | [beshkenadze/lang-id-voxlingua107-ecapa-mlx](https://huggingface.co/beshkenadze/lang-id-voxlingua107-ecapa-mlx) |

### VAD / Speaker Diarization Models

| Model family | Current support in Swift | Model README | Example / default repo |
|--------------|--------------------------|--------------|------------------------|
| Sortformer | Streaming/offline speaker diarization for up to 4 speakers | [Sortformer README](Sources/MLXAudioVAD/Models/Sortformer/README.md) | [mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16](https://huggingface.co/mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16) |
| SmartTurn | Endpoint detection for conversational turn-taking | [SmartTurn README](Sources/MLXAudioVAD/Models/SmartTurn/README.md) | [mlx-community/smart-turn-v3](https://huggingface.co/mlx-community/smart-turn-v3) |

## Features

- **Modular architecture** for minimal app size - import only what you need
- **Automatic model downloading** from HuggingFace Hub
- **Native async/await support** for seamless Swift integration
- **Streaming audio generation** for real-time TTS
- **Type-safe Swift API** with comprehensive error handling
- **Optimized for Apple Silicon** with MLX framework

## Advanced Usage

### Custom Generation Parameters

```swift
let parameters = GenerateParameters(
    maxTokens: 1200,
    temperature: 0.7,
    topP: 0.95,
    repetitionPenalty: 1.5,
    repetitionContextSize: 30
)

let audio = try await model.generate(text: "Your text here", parameters: parameters)
```

### Audio Codec Usage

```swift
import MLXAudioCodecs

// Load SNAC codec
let snac = try await SNAC.fromPretrained("mlx-community/snac_24khz")

// Encode audio to tokens
let tokens = try snac.encode(audio)

// Decode tokens back to audio
let reconstructed = try snac.decode(tokens)
```

### Voice Selection for Multi-Voice Models

```swift
// For models supporting multiple voices (like LlamaTTS/Orpheus)
let audio = try await model.generate(
    text: "Hello!",
    voice: "tara",  // Options: tara, leah, jess, leo, dan, mia, zac, zoe
    parameters: parameters
)
```

## Requirements

- **macOS 14+** or **iOS 17+**
- **Apple Silicon** (M1 or later) recommended for optimal performance
- **Xcode 15+**
- **Swift 5.9+**

## Examples

Check out the [Examples/VoicesApp](Examples/VoicesApp) directory for a complete SwiftUI application demonstrating:
- Loading and running TTS models
- Playing generated audio
- UI components for model interaction

Additional usage examples can be found in the test files.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Credits

- Built on [MLX Swift](https://github.com/ml-explore/mlx-swift)
- Uses [swift-transformers](https://github.com/huggingface/swift-transformers)
- Inspired by [MLX Audio (Python)](https://github.com/Blaizzy/mlx-audio)

## License

MIT License - see [LICENSE](LICENSE) file for details.
