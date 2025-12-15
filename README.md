# Speak2Claude

> Talk to Claude Code with your voice. Just say "Hey Claude" and speak naturally.

## Installation

### Step 1: Download the installer

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/spruikco/speak2claude/main/install.py" -OutFile "install.py"
```

**Mac/Linux:**
```bash
curl -O https://raw.githubusercontent.com/spruikco/speak2claude/main/install.py
```

### Step 2: Run the installer

```bash
python install.py
```

That's it! The installer will:
- Create a virtual environment
- Install all dependencies (PyTorch, Whisper, etc.)
- Download the speech recognition model (~3GB)
- Set up the `/listen` command for Claude Code

### Step 3: Use it

1. Open Claude Code
2. Type `/listen`
3. Say **"Hey Claude"** followed by your message

```
You: "Hey Claude, create a React component for a login form"

[voice] *** WAKE WORD DETECTED! ***
[voice] >>> create a React component for a login form

Claude: I'll create that component for you...
```

---

## Options

### Choose a smaller model (for slower computers)

| Option | Download Size | GPU Memory | Best For |
|--------|--------------|------------|----------|
| `python install.py` | 3 GB | 4 GB | Best accuracy (default) |
| `python install.py --model medium` | 1.5 GB | 2 GB | Good balance |
| `python install.py --model small` | 500 MB | 1 GB | Faster response |
| `python install.py --model tiny` | 150 MB | 500 MB | Low-end hardware |

### No GPU? Use CPU mode

```bash
python install.py --cpu
```

---

## Requirements

- Python 3.8 or newer
- A microphone
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed
- **Recommended:** NVIDIA GPU for faster transcription

---

## Troubleshooting

**"Hey Claude" not working?**
- Speak clearly, pause briefly after "Hey Claude"
- Also works: "Hi Claude", "Okay Claude", "Hey Cloud"

**Slow transcription?**
- Try a smaller model: `python install.py --model small`
- Check GPU is working: run `nvidia-smi`

**No audio input?**
- Check microphone permissions in your OS settings
- Make sure your mic is set as the default recording device

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

## License

MIT - do whatever you want with it.

## Credits

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- Built for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) by Anthropic
