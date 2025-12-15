# Changelog

All notable changes to Speak2Claude will be documented in this file.

## [1.1.0] - 2025-12-15

### Changed
- **Switched to faster-whisper** - Replaced HuggingFace transformers with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for ~4x faster transcription
- **New default model** - Now uses `base.en` instead of `large-v3` for optimal speed/accuracy balance
- **Greedy decoding** - Uses `beam_size=1` for maximum transcription speed
- **VAD filtering** - Enabled Voice Activity Detection to skip silent sections automatically

### Why the change?
The original implementation used `large-v3` with HuggingFace transformers, which provided excellent accuracy but had noticeable latency. After testing, we found that `faster-whisper` with the `base.en` model provides:
- Near-instant transcription (sub-second for most utterances)
- Excellent accuracy for conversational English
- Lower GPU memory usage (~400MB vs ~4GB)

For most voice coding use cases, this is the sweet spot.

### Upgrading
Existing users should reinstall to get the new faster-whisper backend:
```bash
python install.py
```

If you prefer maximum accuracy over speed, you can still use larger models:
```bash
python install.py --model large-v3
```

## [1.0.0] - 2025-12-14

### Added
- Initial release
- Wake word detection ("Hey Claude", "Hi Claude", "Okay Claude")
- OpenAI Whisper speech recognition via HuggingFace transformers
- GPU acceleration with CUDA support
- CPU fallback mode
- Multiple model size options (tiny, small, medium, large-v3)
- Automatic installer with virtual environment setup
- `/listen` slash command for Claude Code integration
