# Speak2Claude

### Talk to Claude Code. Hands-free.

Stop typing. Start talking. **Speak2Claude** lets you control Claude Code with your voice - just say "Hey Claude" and speak naturally.

```
You: "Hey Claude, refactor this function to use async/await"

[voice] *** WAKE WORD DETECTED! ***
[voice] >>> refactor this function to use async/await

Claude: I'll refactor that for you...
```

## Why Speak2Claude?

- **Faster than typing** - Speak your thoughts as fast as you think them
- **Stay in flow** - No context switching between keyboard and mouse
- **Accessibility** - Code without strain on your hands
- **Powered by faster-whisper** - CTranslate2-optimized Whisper for ~4x faster transcription
- **Works offline** - All processing happens locally on your machine
- **GPU accelerated** - Lightning fast on NVIDIA GPUs

---

## Quick Start

### 1. Download

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/spruikco/speak2claude/main/install.py" -OutFile "install.py"
```

**Mac/Linux:**
```bash
curl -O https://raw.githubusercontent.com/spruikco/speak2claude/main/install.py
```

### 2. Install

```bash
python install.py
```

The installer handles everything:
- Creates isolated virtual environment
- Installs PyTorch with CUDA support (auto-detected)
- Downloads Whisper speech recognition model
- Sets up the `/listen` slash command

### 3. Use

1. Open Claude Code
2. Type `/listen`
3. Say **"Hey Claude"** + your command

That's it. You're now voice-controlling Claude.

---

## Model Options

The default `base.en` model offers the best balance of speed and accuracy for English. Choose based on your needs:

| Command | Model | Speed | Accuracy | GPU RAM |
|---------|-------|-------|----------|---------|
| `python install.py` | base.en | Fast | Great | ~400 MB |
| `python install.py --model small.en` | small.en | Moderate | Better | ~1 GB |
| `python install.py --model tiny.en` | tiny.en | Fastest | Good | ~150 MB |
| `python install.py --model large-v3` | large-v3 | Slower | Best | ~3 GB |

**Recommendation:** Start with the default `base.en` - it's fast enough for real-time use and handles conversational English very well.

**No GPU?** Use CPU mode:
```bash
python install.py --cpu
```

---

## Wake Words

Say any of these to activate:
- "Hey Claude"
- "Hi Claude"
- "Okay Claude"
- "Hey Cloud" (it's forgiving!)

---

## Requirements

- Python 3.8+
- Microphone
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed
- NVIDIA GPU recommended (but not required)

---

## Why Not Built-in Dictation?

Your OS has speech recognition built in. Here's why Speak2Claude is better for coding:

| | Speak2Claude | Windows/macOS Dictation |
|---|---|---|
| **Privacy** | 100% local - audio never leaves your machine | Sends audio to Microsoft/Apple servers |
| **Wake word** | "Hey Claude" triggers Claude directly | Just types text - you still hit Enter |
| **Technical accuracy** | Whisper handles code terms, function names, jargon | Optimized for general speech |
| **Offline** | Works without internet | Requires cloud connection |
| **Customizable** | Choose model size, tweak behavior | Take it or leave it |

---

## Troubleshooting

**Wake word not detected?**
- Pause briefly after "Hey Claude"
- Speak clearly into your mic

**Slow transcription?**
- Use a smaller model: `--model small`
- Verify GPU: run `nvidia-smi`

**No audio?**
- Check mic permissions in OS settings
- Set correct default recording device

---

## Uninstall

```bash
# Windows
rmdir /s /q %USERPROFILE%\.speak2claude
del %USERPROFILE%\.claude\commands\listen.md

# Mac/Linux
rm -rf ~/.speak2claude ~/.claude/commands/listen.md
```

---

## Contributing

PRs welcome! Some ideas:
- [ ] Mac/Linux audio feedback
- [ ] Custom wake words
- [ ] Multi-language support
- [ ] Continuous conversation mode

---

## License

MIT - Use it however you want.

---

## Credits

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - CTranslate2-optimized Whisper implementation
- [OpenAI Whisper](https://github.com/openai/whisper) - Original speech recognition model
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) - The AI assistant you're talking to

---

**Built with voice input. Obviously.**
