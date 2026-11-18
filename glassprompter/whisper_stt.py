"""Whisper transcription for the meeting copilot, following OpenWhispr's design (MIT, github.com/OpenWhispr):
a local Whisper model (faster-whisper on CTranslate2, int8 on the CPU) decoding one spoken phrase at a time.

Each audio source gets a Segmenter: an energy gate that follows the room's noise floor and cuts the stream into
phrases at pauses. One decoder thread transcribes finished phrases first and, when it has nothing else to do,
the phrase still being spoken (the live preview). Models download once into the data folder; audio never leaves
the computer.
"""
import logging
import os
import queue
import threading
from collections import deque

import numpy as np

from . import paths

log = logging.getLogger(__name__)

try:
    import faster_whisper
except Exception:                     # optional dependency
    faster_whisper = None

RATE = 16000
BLOCK = 1600                          # 0.1 s
SIZES = {                             # name -> (label, download MB); ".en" models are English-only and more accurate
    "base.en": ("Whisper Base, English (recommended)", 142),
    "small.en": ("Whisper Small, English (most accurate, slower)", 466),
    "tiny.en": ("Whisper Tiny, English (fastest)", 75),
    "base": ("Whisper Base, any language", 142),
    "small": ("Whisper Small, any language", 466),
}
DEFAULT_SIZE = "base.en"
PREFIX = "whisper:"


def available():
    return faster_whisper is not None


def models_dir():
    return paths.ensure(os.path.join(paths.data_dir(), "models", "whisper"))


def is_downloaded(size):
    try:
        from faster_whisper.utils import download_model
        download_model(size, local_files_only=True, cache_dir=models_dir())
        return True
    except Exception:                 # noqa: BLE001 - not in the cache yet
        return False


def load(size):
    """-> WhisperModel. Downloads the model the first time (huggingface.co/Systran)."""
    threads = max(1, min(4, (os.cpu_count() or 2) - 1))
    return faster_whisper.WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=threads,
                                       download_root=models_dir())


def to_16k(pcm, rate):
    """int16 bytes at any rate -> float32 mono at 16 kHz."""
    a = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if rate == RATE or not len(a):
        return a
    n = int(round(len(a) * RATE / rate))
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


class Segmenter:
    """Cuts a 16 kHz stream into phrases. feed() returns a finished phrase (float32 array) or None."""
    END_SILENCE = 0.7                 # seconds of quiet that end a phrase
    MAX_PHRASE = 15.0                 # long monologues are cut so the transcript keeps moving
    MIN_SPEECH = 0.35                 # shorter blips (a cough, a click) are dropped

    def __init__(self):
        self.floor = 0.003
        self.buf = []
        self.pre = deque(maxlen=3)    # a little audio from before the voice started, so first words aren't clipped
        self.quiet = 0
        self.voiced = 0
        self._pending = np.zeros(0, dtype=np.float32)

    @property
    def speaking(self):
        return bool(self.buf)

    def current(self):
        return np.concatenate(self.buf) if self.buf else None

    def feed(self, audio):
        self._pending = np.concatenate([self._pending, audio])
        done = None
        while len(self._pending) >= BLOCK:
            block, self._pending = self._pending[:BLOCK], self._pending[BLOCK:]
            done = self._block(block) if done is None else done
        return done

    def _block(self, block):
        rms = float(np.sqrt(np.mean(block ** 2)))
        loud = rms > max(self.floor * 3.0, 0.004)
        if not self.buf:
            if not loud:
                self.floor = 0.95 * self.floor + 0.05 * rms       # learn the room while nobody talks
                self.pre.append(block)
                return None
            self.buf = list(self.pre) + [block]
            self.quiet, self.voiced = 0, 1
            return None
        self.buf.append(block)
        if loud:
            self.voiced += 1
            self.quiet = 0
        else:
            self.quiet += 1
        seconds = len(self.buf) * BLOCK / RATE
        if self.quiet * BLOCK / RATE >= self.END_SILENCE or seconds >= self.MAX_PHRASE:
            return self.flush()
        return None

    def flush(self):
        phrase, voiced = self.current(), self.voiced
        self.buf, self.quiet, self.voiced = [], 0, 0
        self.pre.clear()
        if phrase is None or voiced * BLOCK / RATE < self.MIN_SPEECH:
            return None
        return phrase


class Decoder:
    """One thread, one model: finished phrases first, then a live preview of whoever is mid-sentence."""
    PREVIEW_EVERY = 1.2               # seconds of new speech between live previews

    def __init__(self, model, language, vocabulary, on_segment, on_partial):
        self.model = model
        self.language = language
        self.prompt = vocabulary.strip() or None      # names and jargon steer the spelling (OpenWhispr's dictionary)
        self.on_segment, self.on_partial = on_segment, on_partial
        self.finals = queue.Queue()
        self.live = {}                # who -> Segmenter
        self._seen = {}               # who -> samples covered by the last preview
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="whisper-decoder", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def sink(self, who, rate):
        seg = Segmenter()
        self.live[who] = seg
        return WhisperSink(self, who, rate, seg)

    def transcribe(self, audio, final=True):
        segments, _ = self.model.transcribe(
            audio, language=self.language, beam_size=3 if final else 1, vad_filter=final,
            condition_on_previous_text=False, initial_prompt=self.prompt, without_timestamps=True)
        words = []
        for s in segments:
            if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:  # Whisper "hearing" words in noise
                continue
            words.append(s.text.strip())
        return " ".join(w for w in words if w).strip()

    def _run(self):
        while not self._stop.is_set():
            try:
                who, audio = self.finals.get(timeout=0.1)
            except queue.Empty:
                self._preview()
                continue
            try:
                text = self.transcribe(audio)
            except Exception as e:                       # noqa: BLE001
                log.warning("Whisper couldn't transcribe a phrase: %s", e)
                text = ""
            self._seen[who] = 0
            self.on_partial(who, "")
            if text:
                self.on_segment(who, text)

    def _preview(self):
        for who, seg in list(self.live.items()):
            audio = seg.current()
            if audio is None or len(audio) - self._seen.get(who, 0) < self.PREVIEW_EVERY * RATE:
                continue
            self._seen[who] = len(audio)
            try:
                text = self.transcribe(audio, final=False)
            except Exception:                            # noqa: BLE001 - a preview is best effort
                continue
            if text and self.finals.empty():
                self.on_partial(who, text)
            return


class WhisperSink:
    """What a capture thread feeds: raw int16 audio in, phrases queued for the decoder."""

    def __init__(self, decoder, who, rate, seg):
        self.decoder, self.who, self.rate, self.seg = decoder, who, rate, seg

    def feed(self, pcm):
        phrase = self.seg.feed(to_16k(pcm, self.rate))
        if phrase is not None:
            self.decoder.finals.put((self.who, phrase))

    def flush(self):
        phrase = self.seg.flush()
        if phrase is not None:
            self.decoder.finals.put((self.who, phrase))
