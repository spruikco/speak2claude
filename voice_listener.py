#!/usr/bin/env python3
"""
Voice Listener - Background process that listens for wake word
and types transcriptions directly into the terminal.

Run this alongside Claude Code for hands-free voice input.
"""

import sys
import time
import warnings
import ctypes

warnings.filterwarnings("ignore")

# Windows API for typing into active window
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

class VoiceListener:
    def __init__(self):
        self.whisper_pipe = None
        self.sample_rate = 16000
        self.silence_threshold = 0.008
        self.silence_duration = 1.8  # Balance responsiveness and completeness
        self.initialized = False
        self.np = None

        # Wake word - "Hey Claude" with many variants
        self.wake_word = "hey claude"
        # All the variants Whisper might transcribe
        self.wake_variants = [
            "hey claude", "hey, claude", "hey claud", "hey clod", "hey cloud",
            "a]claude", "hey clawed", "hey claw", "hey klaud", "hey klaude",
            "hi claude", "hi claud", "hi cloud", "hi clod",
            "okay claude", "ok claude", "o.k. claude",
            "hey, claud", "hey, cloud", "hey, clod",
            "hey clawed,", "heyclod", "heyclaude", "heycloud",
            "hey clout", "hay claude", "hay cloud",
            "a claude", "eh claude", "ey claude",
        ]

    def log(self, msg):
        print(f"[voice] {msg}", flush=True)

    def initialize_whisper(self):
        if self.initialized:
            return True

        self.log("Loading Whisper model (faster-whisper)...")
        try:
            import numpy as np
            from faster_whisper import WhisperModel

            # Check for CUDA
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
                self.log(f"Using device: cuda (GPU accelerated)")
            else:
                device = "cpu"
                compute_type = "int8"
                self.log(f"Using device: cpu")

            # Load faster-whisper model
            self.whisper_model = WhisperModel(
                "large-v3",
                device=device,
                compute_type=compute_type
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
            pass  # Silently fail if beep doesn't work

    def init_audio_stream(self):
        """Initialize persistent audio stream."""
        import sounddevice as sd
        self.chunk_duration = 0.1
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=int(self.sample_rate * self.chunk_duration)
        )
        self.stream.start()
        self.log("Microphone stream initialized (always listening)")

    def record_audio(self):
        """Record audio until silence (using persistent stream)."""
        chunks = []
        silent_time = 0
        speech_time = 0
        min_speech_level = 0.01
        min_speech_duration = 0.3

        start_time = time.time()
        max_time = 30

        while True:
            audio_chunk, _ = self.stream.read(int(self.sample_rate * self.chunk_duration))
            audio_chunk = audio_chunk.flatten()
            chunks.append(audio_chunk)

            level = float(self.np.sqrt(self.np.mean(audio_chunk**2)))
            elapsed = time.time() - start_time

            if level < self.silence_threshold:
                silent_time += self.chunk_duration
            else:
                silent_time = 0
                if level > min_speech_level:
                    speech_time += self.chunk_duration

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
            # faster-whisper transcribe
            segments, info = self.whisper_model.transcribe(
                audio,
                language="en",
                beam_size=1,  # Faster with beam_size=1
                vad_filter=True  # Skip silence
            )
            # Combine all segments
            text = " ".join(segment.text for segment in segments)
            return text.strip()
        except Exception as e:
            self.log(f"Transcription error: {e}")
            return ""

    def detect_wake_word(self, text):
        """Check for wake word 'kangaroo' (with fuzzy matching) and extract message after it."""
        text_lower = text.lower()

        # Check all variants
        for variant in self.wake_variants:
            if variant in text_lower:
                # Extract everything after the wake word variant
                parts = text_lower.split(variant, 1)
                if len(parts) > 1:
                    # Get original case version
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

        # Use clipboard for reliability
        old_clipboard = ""
        try:
            old_clipboard = pyperclip.paste()
        except:
            pass

        pyperclip.copy(text)
        time.sleep(0.05)

        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.05)

        # Press Enter to submit
        pyautogui.press('enter')

        # Restore clipboard
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

        # Initialize persistent audio stream (no more mic startup delay!)
        self.init_audio_stream()

        self.log("=" * 50)
        self.log("VOICE LISTENER ACTIVE")
        self.log(f"Say '{self.wake_word}' followed by your message")
        self.log("Press Ctrl+C to stop")
        self.log("=" * 50)

        while True:
            try:
                # Record
                audio = self.record_audio()

                if audio is None:
                    continue

                # Transcribe
                text = self.transcribe(audio)

                if not text:
                    continue

                # Check for wake word
                wake_detected, message = self.detect_wake_word(text)

                if wake_detected:
                    # Audio beep + visual indicator
                    self.play_beep(1000, 0.15)  # Higher pitch, short beep
                    self.log("*** WAKE WORD DETECTED! ***")
                    if message:
                        self.log(f">>> {message}")
                        self.type_text(message)
                    else:
                        self.log("(wake word only, no message)")
                else:
                    # Show what was heard but not sent
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
