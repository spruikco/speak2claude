# Speak2Claude

> Voice input for Claude Code. Say "Hey Claude" and start talking.

Turn your voice into commands for Claude Code. No more typing long prompts - just speak naturally and watch your words appear in the terminal.

## Demo

```
You: "Hey Claude, create a React component that displays a user profile card with avatar, name, and bio"

[voice] *** WAKE WORD DETECTED! ***
[voice] >>> create a React component that displays a user profile card with avatar, name, and bio

Claude: I'll create that component for you...
```

## Quick Start

### One-Line Install

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/spruikco/speak2claude/main/install.py" -OutFile "install.py"; python install.py
```

**Mac/Linux:**
```bash
curl -sSL https://raw.githubusercontent.com/spruikco/speak2claude/main/install.py | python3
```

### Usage

1. Start Claude Code
2. Type `/listen`
3. Say **"Hey Claude"** followed by your message
4. Your speech is transcribed and sent to Claude

That's it!

## Features

- **Wake Word Detection** - Say "Hey Claude" (or "Hi Claude", "Okay Claude") to activate
- **High Accuracy** - Uses OpenAI's Whisper large-v3 model for best transcription
- **GPU Accelerated** - CUDA support for fast transcription on NVIDIA GPUs
- **Hands-Free** - Keep coding while talking to Claude
- **Easy Install** - One command setup with automatic dependency management

## Installation Options

```bash
# Default: Best quality (requires ~4GB VRAM)
python install.py

# Smaller models for less powerful hardware
python install.py --model medium    # Good quality, ~2GB VRAM
python install.py --model small     # Faster, ~1GB VRAM
python install.py --model tiny      # Fastest, minimal VRAM

# CPU-only mode (no GPU required)
python install.py --cpu
```

## Model Comparison

| Model | Download | Speed | Quality | GPU Memory |
|-------|----------|-------|---------|------------|
| **large-v3** | ~3GB | Slower | Excellent | ~4GB |
| medium | ~1.5GB | Medium | Good | ~2GB |
| small | ~500MB | Fast | Decent | ~1GB |
| tiny | ~150MB | Fastest | Basic | ~500MB |

## Requirements

- Python 3.8+
- Microphone
- [Claude Code](https://claude.ai/code) CLI installed
- **Recommended:** NVIDIA GPU with CUDA for faster transcription

## How It Works

1. A background listener continuously monitors your microphone
2. When you speak, it waits for a pause to know you're finished
3. OpenAI Whisper transcribes your speech to text
4. If "Hey Claude" is detected, your message is typed into the terminal
5. Claude Code processes your request

## Troubleshooting

**Wake word not detected?**
- Speak clearly with a brief pause after "Hey Claude"
- Try alternatives: "Hi Claude", "Okay Claude", "Hey Cloud"

**Transcription is slow?**
- Use a smaller model: `--model small`
- Ensure CUDA is working: `nvidia-smi`

**No audio?**
- Check microphone permissions
- Verify default recording device

## Uninstall

```bash
# Remove all installed files
rm -rf ~/.speak2claude
rm ~/.claude/commands/listen.md
```

## Contributing

Pull requests welcome! Ideas for improvement:
- [ ] Mac audio feedback (currently Windows-only beep)
- [ ] Custom wake word configuration
- [ ] Continuous listening mode without wake word
- [ ] Multi-language support

## License

MIT License - do whatever you want with it.

## Credits

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Hugging Face](https://huggingface.co/) for model hosting
- Built for [Claude Code](https://claude.ai/code) by Anthropic

---

**Made with voice input, obviously.**
