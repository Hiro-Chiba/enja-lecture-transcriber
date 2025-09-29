"""Tkinter-based English speech-to-Japanese translation app."""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk

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
        self.source_text = tk.StringVar(value="")
        self.translation_text = tk.StringVar(value="")

        self._build_ui()

        # Speech/translation backend
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.translator = Translator()

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
        ttk.Label(source_frame, textvariable=self.source_text, wraplength=480, justify=tk.LEFT).pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        translation_frame = ttk.Labelframe(self.root, text="日本語訳")
        translation_frame.pack(fill=tk.BOTH, expand=True, **padding)
        ttk.Label(translation_frame, textvariable=self.translation_text, wraplength=480, justify=tk.LEFT).pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def start_listening(self) -> None:
        if self._running:
            return
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
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
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                text = self.recognizer.recognize_google(audio, language="en-US")
                translation = self.translator.translate(text, src="en", dest="ja").text
                self._queue.put(Transcript(source=text, translation=translation))
            except sr.WaitTimeoutError:
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
                self.source_text.set(item.source)
                self.translation_text.set(item.translation)
                self.status_var.set("翻訳しました")
        except queue.Empty:
            pass
        finally:
            if self._running:
                self.status_var.set("リスニング中…")
            self.root.after(200, self._process_queue)


def main() -> None:
    root = tk.Tk()
    app = SpeechTranslatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
