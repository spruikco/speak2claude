#!/usr/bin/env python3
"""
Speak2Claude Installer
One-click installation for voice input in Claude Code.

Usage:
    Windows: python install.py
    Mac/Linux: python3 install.py

Options:
    --model [large-v3|medium|small|tiny]  Choose Whisper model size (default: large-v3)
    --cpu                                  Force CPU mode (no CUDA)
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path

# ANSI colors for pretty output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_step(msg):
    print(f"{Colors.BLUE}{Colors.BOLD}==>{Colors.END} {msg}")

def print_success(msg):
    print(f"{Colors.GREEN}[OK]{Colors.END} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}[!]{Colors.END} {msg}")

def print_error(msg):
    print(f"{Colors.RED}[ERROR]{Colors.END} {msg}")

def get_install_dir():
    """Get the installation directory based on OS."""
    if sys.platform == "win32":
        return Path.home() / ".speak2claude"
    else:
        return Path.home() / ".speak2claude"

def get_claude_commands_dir():
    """Get Claude Code commands directory."""
    return Path.home() / ".claude" / "commands"

def check_python_version():
    """Ensure Python 3.8+"""
    if sys.version_info < (3, 8):
        print_error("Python 3.8 or higher is required")
        sys.exit(1)
    print_success(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_cuda():
    """Check if CUDA is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        if result.returncode == 0:
            print_success("NVIDIA GPU detected - will use CUDA acceleration")
            return True
    except FileNotFoundError:
        pass
    print_warning("No NVIDIA GPU detected - will use CPU (slower)")
    return False

def create_venv(install_dir):
    """Create a virtual environment."""
    venv_dir = install_dir / "venv"
    if venv_dir.exists():
        print_warning("Virtual environment already exists, skipping...")
        return venv_dir

    print_step("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    print_success("Virtual environment created")
    return venv_dir

def get_pip_path(venv_dir):
    """Get the pip executable path."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "pip.exe"
    else:
        return venv_dir / "bin" / "pip"

def get_python_path(venv_dir):
    """Get the python executable path."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    else:
        return venv_dir / "bin" / "python"

def install_dependencies(venv_dir, use_cuda=True):
    """Install required Python packages."""
    pip = get_pip_path(venv_dir)

    print_step("Upgrading pip...")
    subprocess.run([str(pip), "install", "--upgrade", "pip"], check=True, capture_output=True)

    print_step("Installing PyTorch...")
    if use_cuda:
        # Install CUDA version
        subprocess.run([
            str(pip), "install", "torch", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ], check=True)
    else:
        # Install CPU version
        subprocess.run([str(pip), "install", "torch", "torchaudio"], check=True)
    print_success("PyTorch installed")

    print_step("Installing Whisper and dependencies...")
    packages = [
        "transformers",
        "sounddevice",
        "pyperclip",
        "pyautogui",
        "accelerate"
    ]
    subprocess.run([str(pip), "install"] + packages, check=True)
    print_success("All dependencies installed")

def copy_listener_script(install_dir, model_name="openai/whisper-large-v3"):
    """Copy the voice listener script to install directory."""
    listener_script = '''#!/usr/bin/env python3
"""
Speak2Claude Voice Listener
Say "Hey Claude" followed by your message to speak to Claude Code.
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore")

class VoiceListener:
    def __init__(self, model_name="MODEL_PLACEHOLDER"):
        self.whisper_pipe = None
        self.sample_rate = 16000
        self.silence_threshold = 0.008
        self.silence_duration = 1.8
        self.initialized = False
        self.np = None
        self.model_name = model_name

        # Wake word - "Hey Claude" with many variants
        self.wake_word = "hey claude"
        self.wake_variants = [
            "hey claude", "hey, claude", "hey claud", "hey clod", "hey cloud",
            "hey clawed", "hey claw", "hey klaud", "hey klaude",
            "hi claude", "hi claud", "hi cloud", "hi clod",
            "okay claude", "ok claude", "o.k. claude",
            "hey, claud", "hey, cloud", "hey, clod",
            "heyclod", "heyclaude", "heycloud",
            "hey clout", "hay claude", "hay cloud",
            "a claude", "eh claude", "ey claude",
        ]

    def log(self, msg):
        print(f"[voice] {msg}", flush=True)

    def initialize_whisper(self):
        if self.initialized:
            return True

        self.log("Loading Whisper model...")
        try:
            import torch
            import numpy as np
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            self.log(f"Using device: {device}")

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            model.to(device)

            processor = AutoProcessor.from_pretrained(self.model_name)

            self.whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=dtype,
                device=device,
            )

            self.np = np
            self.initialized = True
            self.log(f"Model loaded! Listening for '{self.wake_word}'...")
            return True

        except Exception as e:
            self.log(f"Failed to load Whisper: {e}")
            return False

    def play_beep(self, frequency=800, duration=0.1):
        """Play a beep sound to indicate listening started."""
        try:
            import winsound
            winsound.Beep(frequency, int(duration * 1000))
        except:
            pass

    def record_audio(self):
        """Record audio until silence."""
        import sounddevice as sd

        chunks = []
        silent_time = 0
        speech_time = 0
        chunk_duration = 0.1
        min_speech_level = 0.01
        min_speech_duration = 0.3

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=int(self.sample_rate * chunk_duration)
        )

        start_time = time.time()
        max_time = 30

        with stream:
            while True:
                audio_chunk, _ = stream.read(int(self.sample_rate * chunk_duration))
                audio_chunk = audio_chunk.flatten()
                chunks.append(audio_chunk)

                level = float(self.np.sqrt(self.np.mean(audio_chunk**2)))
                elapsed = time.time() - start_time

                if level < self.silence_threshold:
                    silent_time += chunk_duration
                else:
                    silent_time = 0
                    if level > min_speech_level:
                        speech_time += chunk_duration

                if silent_time >= self.silence_duration and elapsed > 0.5:
                    break

                if elapsed > max_time:
                    break

        if not chunks or speech_time < min_speech_duration:
            return None

        return self.np.concatenate(chunks)

    def transcribe(self, audio):
        if audio is None or len(audio) < self.sample_rate * 0.3:
            return ""

        try:
            result = self.whisper_pipe(
                {"array": audio, "sampling_rate": self.sample_rate},
                return_timestamps=True,
                generate_kwargs={"task": "transcribe", "language": "en"}
            )
            return result["text"].strip()
        except Exception as e:
            self.log(f"Transcription error: {e}")
            return ""

    def detect_wake_word(self, text):
        """Check for wake word and extract message after it."""
        text_lower = text.lower()

        for variant in self.wake_variants:
            if variant in text_lower:
                parts = text_lower.split(variant, 1)
                if len(parts) > 1:
                    wake_pos = text_lower.find(variant)
                    message = text[wake_pos + len(variant):].strip()
                    message = message.lstrip(",.!? ")
                    return True, message if message else text.strip()
                return True, ""

        return False, ""

    def type_text(self, text):
        """Type text into the active window."""
        if not text:
            return

        import pyautogui
        import pyperclip

        old_clipboard = ""
        try:
            old_clipboard = pyperclip.paste()
        except:
            pass

        pyperclip.copy(text)
        time.sleep(0.05)

        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.05)

        pyautogui.press('enter')

        try:
            if old_clipboard:
                time.sleep(0.1)
                pyperclip.copy(old_clipboard)
        except:
            pass

    def run(self):
        """Main loop - continuously listen for wake word."""
        if not self.initialize_whisper():
            return

        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.01

        self.log("=" * 50)
        self.log("VOICE LISTENER ACTIVE")
        self.log(f"Say '{self.wake_word}' followed by your message")
        self.log("Press Ctrl+C to stop")
        self.log("=" * 50)

        while True:
            try:
                audio = self.record_audio()

                if audio is None:
                    continue

                text = self.transcribe(audio)

                if not text:
                    continue

                wake_detected, message = self.detect_wake_word(text)

                if wake_detected:
                    self.play_beep(1000, 0.15)
                    self.log("*** WAKE WORD DETECTED! ***")
                    if message:
                        self.log(f">>> {message}")
                        self.type_text(message)
                    else:
                        self.log("(wake word only, no message)")
                else:
                    self.log(f"[ignored] {text[:50]}...")

            except KeyboardInterrupt:
                self.log("Stopping...")
                break
            except Exception as e:
                self.log(f"Error: {e}")
                time.sleep(1)


if __name__ == "__main__":
    listener = VoiceListener()
    listener.run()
'''.replace("MODEL_PLACEHOLDER", model_name)

    script_path = install_dir / "voice_listener.py"
    script_path.write_text(listener_script)
    print_success("Voice listener script installed")

def install_slash_command(install_dir):
    """Install the /listen slash command for Claude Code."""
    commands_dir = get_claude_commands_dir()
    commands_dir.mkdir(parents=True, exist_ok=True)

    python_path = get_python_path(install_dir / "venv")
    listener_path = install_dir / "voice_listener.py"

    command_content = f'''Start voice listener for hands-free input. Say "Hey Claude" followed by your message.

Run this command in background:
```bash
"{python_path}" "{listener_path}"
```

The listener will continuously listen for "Hey Claude" and type your speech into the terminal.
Press Ctrl+C in the background terminal to stop.
'''

    command_path = commands_dir / "listen.md"
    command_path.write_text(command_content)
    print_success(f"Slash command installed: /listen")

def predownload_model(venv_dir, model_name):
    """Pre-download the Whisper model."""
    python = get_python_path(venv_dir)

    print_step(f"Downloading Whisper model ({model_name})...")
    print_warning("This may take a few minutes for the first download (~3GB for large-v3)")

    download_script = f'''
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
print("Downloading model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained("{model_name}", low_cpu_mem_usage=True, use_safetensors=True)
processor = AutoProcessor.from_pretrained("{model_name}")
print("Model downloaded successfully!")
'''

    result = subprocess.run(
        [str(python), "-c", download_script],
        capture_output=False
    )

    if result.returncode == 0:
        print_success("Whisper model downloaded and cached")
    else:
        print_warning("Model will be downloaded on first use")

def main():
    parser = argparse.ArgumentParser(description="Install Speak2Claude")
    parser.add_argument(
        "--model",
        choices=["large-v3", "medium", "small", "tiny"],
        default="large-v3",
        help="Whisper model size (default: large-v3, best quality)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (no CUDA)"
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Skip pre-downloading the model"
    )
    args = parser.parse_args()

    # Model name mapping
    model_map = {
        "large-v3": "openai/whisper-large-v3",
        "medium": "openai/whisper-medium",
        "small": "openai/whisper-small",
        "tiny": "openai/whisper-tiny"
    }
    model_name = model_map[args.model]

    print()
    print(f"{Colors.BOLD}=== Speak2Claude Installer ==={Colors.END}")
    print(f"Voice input for Claude Code")
    print()

    # Check requirements
    check_python_version()
    use_cuda = not args.cpu and check_cuda()

    # Setup directories
    install_dir = get_install_dir()
    install_dir.mkdir(parents=True, exist_ok=True)
    print_success(f"Install directory: {install_dir}")

    # Create venv and install dependencies
    venv_dir = create_venv(install_dir)
    install_dependencies(venv_dir, use_cuda)

    # Copy scripts
    copy_listener_script(install_dir, model_name)

    # Install slash command
    install_slash_command(install_dir)

    # Download model
    if not args.skip_model_download:
        predownload_model(venv_dir, model_name)

    print()
    print(f"{Colors.GREEN}{Colors.BOLD}=== Installation Complete! ==={Colors.END}")
    print()
    print("To use voice input in Claude Code:")
    print(f"  1. Open Claude Code in your terminal")
    print(f"  2. Type: {Colors.BOLD}/listen{Colors.END}")
    print(f"  3. Say: {Colors.BOLD}Hey Claude, <your message>{Colors.END}")
    print()
    print(f"Manual start: {get_python_path(venv_dir)} {install_dir / 'voice_listener.py'}")
    print()

if __name__ == "__main__":
    main()
