"""Tkinter-based English speech-to-Japanese translation app."""
from __future__ import annotations

import math
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, scrolledtext, ttk

import speech_recognition as sr
from googletrans import Translator


@dataclass
class Transcript:
    source: str
    translation: str


class SpeechTranslatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("英語→日本語 リアルタイム翻訳")

        # UI components
        self.status_var = tk.StringVar(value="待機中")

        self.source_log: list[str] = []
        self.translation_log: list[str] = []

        self._build_ui()

        # Speech/translation backend
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.translator = Translator()

        # Fine-tune recognizer for better accuracy
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 300
        self.recognizer.pause_threshold = 0.6
        self.recognizer.non_speaking_duration = 0.3

        # Runtime calibration
        self._last_calibration = 0.0

        # Try multiple accent-specific language codes when recognizing speech
        self.recognition_languages = [
            "en-US",
            "en-GB",
            "en-IN",
            "en-AU",
        ]

        # Threading
        self._queue: "queue.Queue[Transcript | Exception]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self._running = False

        # Periodic UI update
        self.root.after(200, self._process_queue)

    def _build_ui(self) -> None:
        padding = {"padx": 10, "pady": 5}

        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, **padding)
        ttk.Label(status_frame, text="状態:").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)

        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, **padding)
        self.start_button = ttk.Button(button_frame, text="開始", command=self.start_listening)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        self.stop_button = ttk.Button(button_frame, text="停止", command=self.stop_listening, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)

        source_frame = ttk.Labelframe(self.root, text="英語の認識結果")
        source_frame.pack(fill=tk.BOTH, expand=True, **padding)
        self.source_text_widget = scrolledtext.ScrolledText(
            source_frame,
            wrap=tk.WORD,
            height=10,
            state=tk.DISABLED,
        )
        self.source_text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        translation_frame = ttk.Labelframe(self.root, text="日本語訳")
        translation_frame.pack(fill=tk.BOTH, expand=True, **padding)
        self.translation_text_widget = scrolledtext.ScrolledText(
            translation_frame,
            wrap=tk.WORD,
            height=10,
            state=tk.DISABLED,
        )
        self.translation_text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def start_listening(self) -> None:
        if self._running:
            return
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self._last_calibration = time.time()
        except OSError as exc:
            messagebox.showerror("マイクエラー", f"マイクにアクセスできません: {exc}")
            return

        self._running = True
        self.status_var.set("リスニング中…")
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)

        self._worker = threading.Thread(target=self._listen_loop, daemon=True)
        self._worker.start()

    def stop_listening(self) -> None:
        if not self._running:
            return
        self._running = False
        self.status_var.set("停止中…")
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)

    def _listen_loop(self) -> None:
        while self._running:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(
                        source,
                        timeout=5,
                        phrase_time_limit=6,
                    )

                if self._should_recalibrate(audio):
                    self._calibrate_noise()
                    continue

                text = self._recognize_with_accents(audio)
                if not text:
                    continue

                translation = self.translator.translate(text, src="en", dest="ja").text
                if not translation:
                    continue

                self._queue.put(Transcript(source=text, translation=translation))
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                self._calibrate_noise(longer=True)
                continue
            except Exception as exc:  # pylint: disable=broad-except
                self._queue.put(exc)
                time.sleep(1)

    def _process_queue(self) -> None:
        try:
            while True:
                item = self._queue.get_nowait()
                if isinstance(item, Exception):
                    messagebox.showerror("エラー", str(item))
                    self.stop_listening()
                    break
                self._append_transcript(item)
                self.status_var.set("翻訳しました")
        except queue.Empty:
            pass
        finally:
            if self._running:
                self.status_var.set("リスニング中…")
            self.root.after(200, self._process_queue)

    def _recognize_with_accents(self, audio: sr.AudioData) -> str:
        """Try multiple English variants to better handle accented speech."""
        last_error: Exception | None = None
        best_confidence = -math.inf
        best_transcript = ""
        for language_code in self.recognition_languages:
            try:
                result = self.recognizer.recognize_google(
                    audio,
                    language=language_code,
                    show_all=True,
                )
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                continue

            if isinstance(result, dict):
                alternatives = result.get("alternative", [])
                if alternatives:
                    for alternative in alternatives:
                        transcript = alternative.get("transcript", "").strip()
                        confidence = alternative.get("confidence")
                        if transcript:
                            if confidence is None:
                                confidence = 0.6  # assume reasonable default
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_transcript = transcript
            elif isinstance(result, str) and result:
                return result.strip()

        if best_transcript:
            return best_transcript

        if last_error is not None:
            raise last_error
        raise sr.UnknownValueError("音声を認識できませんでした")

    def _append_transcript(self, transcript: Transcript) -> None:
        self.source_log.append(transcript.source)
        self.translation_log.append(transcript.translation)

        self._append_text(self.source_text_widget, transcript.source)
        self._append_text(self.translation_text_widget, transcript.translation)

    def _append_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        if widget.index("end-1c") != "1.0":
            widget.insert(tk.END, "\n\n")
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def _should_recalibrate(self, audio: sr.AudioData) -> bool:
        """Check whether the captured audio is too quiet or silent."""
        rms = self._calculate_rms(audio)
        if rms < 50:  # empirically chosen threshold for low-volume audio
            return True
        return False

    def _calculate_rms(self, audio: sr.AudioData) -> float:
        raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
        if not raw:
            return 0.0

        sample_count = len(raw) // 2
        if sample_count == 0:
            return 0.0

        sum_squares = 0.0
        for i in range(0, len(raw), 2):
            sample = int.from_bytes(raw[i : i + 2], byteorder="little", signed=True)
            sum_squares += sample * sample

        return math.sqrt(sum_squares / sample_count)

    def _calibrate_noise(self, longer: bool = False) -> None:
        """Re-calibrate ambient noise levels to avoid recognition errors."""
        # Limit calibration frequency to avoid blocking the UI.
        now = time.time()
        if now - self._last_calibration < 5:
            return

        duration = 1.5 if longer else 0.8
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)
        except OSError:
            return
        self._last_calibration = now


def main() -> None:
    root = tk.Tk()
    app = SpeechTranslatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
