"""Tkinter製の英語音声→日本語翻訳アプリ。"""
from __future__ import annotations

import collections
import html
import math
import queue
import re
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, scrolledtext
from typing import Any

import speech_recognition as sr
from googletrans import Translator
import webrtcvad

try:  # Google Cloud Translation API（任意）
    from google.cloud import translate_v2 as google_translate_v2
except Exception:  # pragma: no cover - 任意依存関係が存在しない場合がある
    google_translate_v2 = None


ACCENT_LABELS = {
    "en-US": "アメリカ英語",
    "en-GB": "イギリス英語",
    "en-IN": "インド英語",
    "en-AU": "オーストラリア英語",
}

DEFAULT_RECOGNITION_LANGUAGES = list(ACCENT_LABELS)


@dataclass
class Transcript:
    """UIで扱いやすい認識・翻訳結果。"""

    source: str
    translation: str
    accent_code: str
    accent_label: str
    confidence: float
    timestamp: float


class SpeechTranslatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("英語→日本語 リアルタイム翻訳")

        self.status_var = tk.StringVar(value="待機中")
        self.accent_var = tk.StringVar(value="アクセント: -")
        self.confidence_var = tk.StringVar(value="信頼度: -")

        self.source_log: list[str] = []
        self.translation_log: list[str] = []

        self._build_ui()
        self._configure_recognizer()
        self._configure_vad()

        self.translation_helper = GoogleTranslateHelper()
        self.recognition_languages = list(DEFAULT_RECOGNITION_LANGUAGES)

        self._queue: "queue.Queue[Transcript | Exception]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self._running = False
        self._last_calibration = 0.0

        self.root.after(200, self._process_queue)

    def _configure_recognizer(self) -> None:
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(sample_rate=16000, chunk_size=480)
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 300
        self.recognizer.pause_threshold = 0.6
        self.recognizer.non_speaking_duration = 0.3

    def _configure_vad(self) -> None:
        self.vad = webrtcvad.Vad(2)
        self._frame_duration_ms = 30
        self._padding_duration_ms = 300
        self._max_segment_ms = 9000
        self._max_initial_silence_ms = 2000

    def _build_ui(self) -> None:
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        default_width = min(1000, screen_width, 1920)
        default_height = min(780, screen_height, 1200)
        self.root.geometry(f"{int(default_width)}x{int(default_height)}")
        self.root.maxsize(1920, 1200)

        main_frame = tk.Frame(self.root, padx=12, pady=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        header = tk.Frame(main_frame)
        header.pack(fill=tk.X)
        title = tk.Label(header, text="英語→日本語 リアルタイム翻訳", anchor="w")
        title.pack(fill=tk.X)
        subtitle = tk.Label(header, text="英語の音声をシンプルな画面で翻訳します", anchor="w")
        subtitle.pack(fill=tk.X, pady=(2, 10))

        status_frame = tk.Frame(main_frame)
        status_frame.pack(fill=tk.X)
        self.status_label = tk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padx=8, pady=4)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._default_status_bg = self.status_label.cget("bg")

        accent_info = tk.Frame(status_frame)
        accent_info.pack(side=tk.RIGHT, anchor="e")
        accent_label = tk.Label(accent_info, textvariable=self.accent_var, anchor="e")
        accent_label.pack(fill=tk.X)
        confidence_label = tk.Label(accent_info, textvariable=self.confidence_var, anchor="e")
        confidence_label.pack(fill=tk.X)

        button_frame = tk.Frame(main_frame, pady=8)
        button_frame.pack(fill=tk.X)
        self.start_button = tk.Button(button_frame, text="開始", command=self.start_listening)
        self.start_button.pack(side=tk.LEFT, padx=(0, 6))
        self.stop_button = tk.Button(button_frame, text="停止", command=self.stop_listening, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)
        clear_button = tk.Button(button_frame, text="クリア", command=self.clear_logs)
        clear_button.pack(side=tk.RIGHT)

        helper_card = tk.LabelFrame(main_frame, text="使い方")
        helper_card.pack(fill=tk.X, pady=(0, 12))
        helper_text = tk.Label(
            helper_card,
            text="静かな場所で利用し、止まったら停止→開始で立て直せます。",
            justify=tk.LEFT,
            wraplength=max(default_width - 80, 200),
        )
        helper_text.pack(fill=tk.X, padx=8, pady=6)

        source_frame = tk.LabelFrame(main_frame, text="英語の認識結果")
        source_frame.pack(fill=tk.BOTH, expand=True)
        self.source_text_widget = scrolledtext.ScrolledText(
            source_frame,
            wrap=tk.WORD,
            height=10,
            state=tk.DISABLED,
        )
        self.source_text_widget.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        translation_frame = tk.LabelFrame(main_frame, text="日本語訳")
        translation_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.translation_text_widget = scrolledtext.ScrolledText(
            translation_frame,
            wrap=tk.WORD,
            height=10,
            state=tk.DISABLED,
        )
        self.translation_text_widget.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.source_text_widget.tag_configure("timestamp", font=("TkDefaultFont", 9, "bold"))
        self.source_text_widget.tag_configure("accent", font=("TkDefaultFont", 9, "bold"))
        self.translation_text_widget.tag_configure("timestamp", font=("TkDefaultFont", 9, "bold"))
        self.translation_text_widget.tag_configure("translation", font=("TkDefaultFont", 10))

        self._update_status("待機中")

    def start_listening(self) -> None:
        if self._running:
            return
        self._update_status("マイクを準備中…")
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self._last_calibration = time.time()
        except OSError as exc:
            messagebox.showerror("マイクエラー", f"マイクにアクセスできません: {exc}")
            self._update_status("マイクエラー")
            return

        self._running = True
        self._update_status("リスニング中…")
        self._set_controls_running(True)

        self._worker = threading.Thread(target=self._listen_loop, daemon=True)
        self._worker.start()

    def stop_listening(self) -> None:
        if not self._running:
            return
        self._running = False
        self._update_status("停止中…")
        self._set_controls_running(False)

    def _set_controls_running(self, running: bool) -> None:
        self.start_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _listen_loop(self) -> None:
        while self._running:
            try:
                with self.microphone as source:
                    audio = self._record_segment(source)

                if audio is None:
                    time.sleep(0.05)
                    continue

                if self._should_recalibrate(audio):
                    self._calibrate_noise()
                    continue

                text, accent_code, confidence = self._recognize_with_accents(audio)
                if not text:
                    continue

                translation = self.translation_helper.translate(text)
                if not translation:
                    continue

                transcript = Transcript(
                    source=text,
                    translation=translation,
                    accent_code=accent_code,
                    accent_label=self._format_accent(accent_code),
                    confidence=confidence,
                    timestamp=time.time(),
                )
                self._queue.put(transcript)
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
                self._update_status("翻訳しました")
        except queue.Empty:
            pass
        finally:
            if self._running:
                self._update_status("リスニング中…")
            self.root.after(200, self._process_queue)

    def _recognize_with_accents(self, audio: sr.AudioData) -> tuple[str, str, float]:
        """アクセントの違いに対応するため複数の英語設定で認識を試す。"""
        last_error: Exception | None = None
        best_confidence = -math.inf
        best_transcript = ""
        best_language = ""
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
                                confidence = 0.6  # 妥当な既定値を仮定
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_transcript = transcript
                                best_language = language_code
            elif isinstance(result, str) and result:
                return result.strip(), language_code, 0.6

        if best_transcript:
            confidence = best_confidence if best_confidence > -math.inf else 0.6
            return best_transcript, best_language or self.recognition_languages[0], confidence

        if last_error is not None:
            raise last_error
        raise sr.UnknownValueError("音声を認識できませんでした")

    def _append_transcript(self, transcript: Transcript) -> None:
        self.source_log.append(transcript.source)
        self.translation_log.append(transcript.translation)

        timestamp = time.strftime("%H:%M:%S", time.localtime(transcript.timestamp))
        self._append_source_entry(timestamp, transcript)
        self._append_translation_entry(timestamp, transcript.translation)
        self.accent_var.set(f"アクセント: {transcript.accent_label}")
        if transcript.confidence > 0:
            self.confidence_var.set(f"信頼度: {transcript.confidence * 100:.1f}%")
        else:
            self.confidence_var.set("信頼度: -")

    def _append_source_entry(self, timestamp: str, transcript: Transcript) -> None:
        widget = self.source_text_widget
        widget.configure(state=tk.NORMAL)
        if widget.index("end-1c") != "1.0":
            widget.insert(tk.END, "\n", ("divider",))
        widget.insert(tk.END, f"{timestamp} ", ("timestamp",))
        widget.insert(widget.index(tk.END), f"[{transcript.accent_label}]\n", ("accent",))
        widget.insert(tk.END, transcript.source)
        widget.insert(tk.END, "\n")
        widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def _append_translation_entry(self, timestamp: str, text: str) -> None:
        widget = self.translation_text_widget
        widget.configure(state=tk.NORMAL)
        if widget.index("end-1c") != "1.0":
            widget.insert(tk.END, "\n", ("divider",))
        widget.insert(tk.END, f"{timestamp}\n", ("timestamp",))
        widget.insert(tk.END, text, ("translation",))
        widget.insert(tk.END, "\n")
        widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def clear_logs(self) -> None:
        self.source_log.clear()
        self.translation_log.clear()
        self.source_text_widget.configure(state=tk.NORMAL)
        self.source_text_widget.delete("1.0", tk.END)
        self.source_text_widget.configure(state=tk.DISABLED)
        self.translation_text_widget.configure(state=tk.NORMAL)
        self.translation_text_widget.delete("1.0", tk.END)
        self.translation_text_widget.configure(state=tk.DISABLED)
        self.accent_var.set("アクセント: -")
        self.confidence_var.set("信頼度: -")
        self._update_status("履歴をクリアしました")

    def _should_recalibrate(self, audio: sr.AudioData) -> bool:
        """取得した音声が小さすぎるか無音かを確認する。"""
        return self._calculate_rms(audio) < 50  # 小さな音量を判定する経験的なしきい値

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
        """周囲のノイズレベルを再調整して認識エラーを防ぐ。"""
        # UIが止まらないように再キャリブレーションの頻度を制限する。
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

    def _record_segment(self, source: sr.AudioSource) -> sr.AudioData | None:
        """WebRTC VADを使って堅牢に音声区間を切り出す。"""
        sample_rate = source.SAMPLE_RATE or self.microphone.SAMPLE_RATE
        sample_width = source.SAMPLE_WIDTH or self.microphone.SAMPLE_WIDTH
        if not sample_rate or not sample_width:
            return None

        frame_length = int(sample_rate * self._frame_duration_ms / 1000)
        if frame_length <= 0:
            return None

        padding_frames = max(1, self._padding_duration_ms // self._frame_duration_ms)
        max_silence_frames = max(1, self._max_initial_silence_ms // self._frame_duration_ms)
        max_segment_frames = max(1, self._max_segment_ms // self._frame_duration_ms)

        ring_buffer: "collections.deque[tuple[bytes, bool]]" = collections.deque(maxlen=padding_frames)
        voiced_frames: list[bytes] = []
        triggered = False
        initial_silence_frames = 0

        while self._running:
            try:
                frame = source.stream.read(frame_length, exception_on_overflow=False)
            except OSError:
                return None

            if len(frame) < frame_length * sample_width:
                continue

            try:
                is_speech = self.vad.is_speech(frame, sample_rate)
            except ValueError:
                continue

            if not triggered:
                ring_buffer.append((frame, is_speech))
                if is_speech:
                    initial_silence_frames = 0
                else:
                    initial_silence_frames += 1

                num_voiced = sum(1 for _, speech in ring_buffer if speech)
                if len(ring_buffer) == ring_buffer.maxlen and num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    voiced_frames.extend(f for f, _ in ring_buffer)
                    ring_buffer.clear()
                    continue

                if initial_silence_frames >= max_silence_frames:
                    return None
            else:
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))

                num_unvoiced = sum(1 for _, speech in ring_buffer if not speech)
                if (
                    (len(ring_buffer) == ring_buffer.maxlen and num_unvoiced > 0.85 * ring_buffer.maxlen)
                    or len(voiced_frames) >= max_segment_frames
                ):
                    break

        if not voiced_frames:
            return None

        raw_audio = b"".join(voiced_frames)
        return sr.AudioData(raw_audio, sample_rate, sample_width)

    def _update_status(self, text: str, color: str | None = None) -> None:
        self.status_var.set(text)
        if color:
            self.status_label.configure(bg=color)
        else:
            self.status_label.configure(bg=self._default_status_bg)

    def _format_accent(self, language_code: str) -> str:
        return ACCENT_LABELS.get(language_code, f"{language_code} アクセント")


class GoogleTranslateHelper:
    """公式APIとgoogletransをまとめた翻訳ヘルパー。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, str] = {}
        self._official_client = self._create_official_client()
        self._translator = Translator()

    def _create_official_client(self) -> Any | None:
        if google_translate_v2 is None:
            return None

        try:
            return google_translate_v2.Client()
        except Exception:
            return None

    def translate(self, text: str, dest: str = "ja", src: str = "en") -> str:
        normalized = text.strip()
        if not normalized:
            return ""

        with self._lock:
            cached = self._cache.get(normalized)
        if cached is not None:
            return cached

        translation = self._translate_official(normalized, dest=dest, src=src)
        if not translation:
            translation = self._translate_with_strategies(normalized, dest=dest, src=src)

        if translation:
            with self._lock:
                self._cache[normalized] = translation
        return translation

    def _translate_official(self, text: str, dest: str, src: str) -> str:
        if self._official_client is None:
            return ""

        request: dict[str, Any] = {"target_language": dest, "format_": "text"}
        if src != "auto":
            request["source_language"] = src

        try:
            result = self._official_client.translate(text, **request)
        except Exception:
            return ""

        if isinstance(result, list):
            translations = [self._normalize_official(item.get("translatedText", "")) for item in result]
            joined = "\n".join(filter(None, translations)).strip()
            return joined

        if isinstance(result, dict):
            translated = self._normalize_official(result.get("translatedText", ""))
            return translated

        return ""

    def _normalize_official(self, text: str) -> str:
        return html.unescape(text.strip()) if text else ""

    def _translate_with_strategies(self, text: str, dest: str, src: str) -> str:
        strategies = (
            {"src": src, "dest": dest},
            {"src": "auto", "dest": dest},
        )

        last_error: Exception | None = None
        for strategy in strategies:
            try:
                result = self._translator.translate(text, **strategy)
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                continue

            translated = result.text.strip()
            if translated:
                return translated

        segmented_translation = self._translate_segmented(text, dest=dest, src=src)
        if segmented_translation:
            return segmented_translation

        if last_error is not None:
            raise last_error
        return ""

    def _translate_segmented(self, text: str, dest: str, src: str) -> str:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return ""

        translated_sentences: list[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            try:
                result = self._translator.translate(sentence, src=src, dest=dest)
            except Exception:  # pylint: disable=broad-except
                continue
            translated = result.text.strip()
            if translated:
                translated_sentences.append(translated)

        return "\n".join(translated_sentences).strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [part for part in parts if part]


def main() -> None:
    root = tk.Tk()
    app = SpeechTranslatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
