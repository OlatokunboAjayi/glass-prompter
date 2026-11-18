"""Meeting copilot audio: your microphone ("You") and the computer's sound ("Them") -> offline live transcript.

Speech engine: local Whisper (whisper_stt.py) when faster-whisper is installed, else Vosk. Two capture threads,
each feeding its own sink (a Whisper phrase segmenter, or a Vosk recognizer on a shared model):
  * microphone through sounddevice (same device as Voice Follow)
  * system audio (what the other people say on Zoom, Teams, Meet...) through a WASAPI loopback on Windows or a
    PulseAudio monitor on Linux, via the optional `soundcard` package. macOS has no loopback without a virtual audio
    device (e.g. BlackHole); there the copilot hears your microphone only, unless you pick such a device.

Both engines run on the computer; nothing leaves it.
"""
import json
import logging
import queue
import threading
import time
import warnings

from PySide6.QtCore import QObject, Signal

from . import paths, whisper_stt
from . import platform as native
from .meeting import ME, THEM

log = logging.getLogger(__name__)

try:
    import vosk
    vosk.SetLogLevel(-1)
except Exception:                     # optional dependency
    vosk = None
try:
    import sounddevice as sd
except Exception:
    sd = None

RATE = 16000
BLOCK_SEC = 0.1


def system_audio_supported():
    """Can we hear the computer's own sound (the other side of a call)?"""
    if native.OS == "macos":
        return False
    try:
        import soundcard  # noqa: F401
        return True
    except Exception:                                    # noqa: BLE001 - missing package or no audio server
        return False


def model_path(choice=""):
    models = paths.speech_models()
    if choice and choice in models:
        return models[choice]
    return paths.model_dir() or (next(iter(models.values())) if models else None)


def engine(choice=""):
    """Setting -> ("whisper", size) or ("vosk", model folder). "" picks Whisper when it is installed."""
    if choice.startswith(whisper_stt.PREFIX) or (not choice and whisper_stt.available()):
        size = choice[len(whisper_stt.PREFIX):] if choice else whisper_stt.DEFAULT_SIZE
        if whisper_stt.available() and size in whisper_stt.SIZES:
            return ("whisper", size)
        return ("vosk", model_path(""))
    return ("vosk", model_path(choice))


class _VoskSink:
    """A Vosk recognizer that emits finished phrases and the words heard so far."""

    def __init__(self, listener, model, who, rate):
        self.l, self.who, self.last = listener, who, ""
        self.rec = vosk.KaldiRecognizer(model, rate)

    def feed(self, data):
        if self.rec.AcceptWaveform(data):
            text = json.loads(self.rec.Result()).get("text", "").strip()
            if text:
                self.l.segment.emit(self.who, text)
            self.l.partial.emit(self.who, "")
            self.last = ""
            return
        text = json.loads(self.rec.PartialResult()).get("partial", "").strip()
        if text != self.last:
            self.l.partial.emit(self.who, text)
            self.last = text

    def flush(self):
        try:
            text = json.loads(self.rec.FinalResult()).get("text", "").strip()
            if text:
                self.l.segment.emit(self.who, text)
        except Exception:                                # noqa: BLE001
            pass
        self.l.partial.emit(self.who, "")


class MeetingListener(QObject):
    segment = Signal(str, str)        # (who, finished phrase)
    partial = Signal(str, str)        # (who, words so far in the current phrase)
    status = Signal(str)              # "loading" | "listening" | "stopped"
    note = Signal(str)                # something the user should know, not fatal (e.g. no system audio)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self._stop = threading.Event()
        self._threads = []
        self._models = {}
        self._lock = threading.Lock()
        self._decoder = None
        self.listening = False
        self.engine_name = ""

    @staticmethod
    def available(choice=""):
        kind, where = engine(choice)
        return kind == "whisper" or (vosk is not None and where is not None)

    @staticmethod
    def why_unavailable(choice=""):
        if MeetingListener.available(choice):
            return ""
        if vosk is None:
            return "Speech recognition isn't part of this install."
        return "No speech model found. See Settings › Voice Follow."

    def _model(self, key, make):
        with self._lock:
            if key not in self._models:
                self._models[key] = make()
            return self._models[key]

    def start(self, mic_device="", system_audio=True, model_choice="", vocabulary=""):
        if self.listening:
            return
        if not self.available(model_choice):
            self.failed.emit(self.why_unavailable(model_choice))
            return
        self._stop.clear()
        self.listening = True
        self.status.emit("loading")

        def boot():
            kind, where = engine(model_choice)
            make_sink = None
            if kind == "whisper":
                try:
                    if not whisper_stt.is_downloaded(where):
                        self.note.emit("Downloading the speech model (%d MB). This happens once."
                                       % whisper_stt.SIZES[where][1])
                    model = self._model(("whisper", where), lambda: whisper_stt.load(where))
                    lang = "en" if where.endswith(".en") else None
                    self._decoder = whisper_stt.Decoder(model, lang, vocabulary, self.segment.emit, self.partial.emit)
                    self._decoder.start()
                    make_sink = self._decoder.sink
                    self.engine_name = whisper_stt.SIZES[where][0]
                except Exception as e:                   # noqa: BLE001 - offline on first use, disk full...
                    log.warning("Whisper unavailable, using Vosk: %s", e)
                    kind, where = "vosk", model_path("")
                    if vosk is None or where is None:
                        self.listening = False
                        self.failed.emit("Couldn't load the speech model: %s" % e)
                        self.status.emit("stopped")
                        return
                    self.note.emit("Couldn't load Whisper (%s). Using the basic offline engine." % e)
            if kind == "vosk":
                try:
                    model = self._model(where, lambda: vosk.Model(where))
                except Exception as e:                   # noqa: BLE001
                    self.listening = False
                    self.failed.emit("Couldn't load the speech model: %s" % e)
                    self.status.emit("stopped")
                    return
                make_sink = lambda who, rate: _VoskSink(self, model, who, rate)  # noqa: E731
                self.engine_name = "Vosk (basic, offline)"
            if self._stop.is_set():
                return
            runners = []
            if sd is not None:
                runners.append((self._run_mic, (make_sink, mic_device)))
            if system_audio:
                if system_audio_supported():
                    runners.append((self._run_system, (make_sink,)))
                elif native.OS == "macos":
                    self.note.emit("macOS can't share the call's sound with apps, so only your microphone is "
                                   "transcribed. Use headphones off, or a loopback device like BlackHole.")
            if not runners:
                self.listening = False
                self.failed.emit("No audio input is available.")
                self.status.emit("stopped")
                return
            for fn, args in runners:
                t = threading.Thread(target=fn, args=args, name="meeting-" + fn.__name__, daemon=True)
                self._threads.append(t)
                t.start()
            self.status.emit("listening")
        threading.Thread(target=boot, name="meeting-boot", daemon=True).start()

    def stop(self):
        self._stop.set()
        self.listening = False
        self._threads = []
        if self._decoder is not None:
            self._decoder.stop()                         # finishes the last phrases, then exits
            self._decoder = None
        self.status.emit("stopped")

    # ---------------------------------------------------------------- recognition
    def _run_mic(self, make_sink, device_name):
        from .voice import _device_index, pick_rate
        try:
            device = _device_index(device_name)
            rate = pick_rate(device)
            sink = make_sink(ME, rate)
            audio = queue.Queue(maxsize=400)

            def callback(indata, frames, t, st):
                try:
                    audio.put_nowait(bytes(indata))
                except queue.Full:
                    pass
            with sd.RawInputStream(samplerate=rate, blocksize=int(rate * BLOCK_SEC), dtype="int16", channels=1,
                                   device=device, callback=callback):
                log.info("Meeting copilot hearing the microphone (%d Hz)", rate)
                while not self._stop.is_set():
                    try:
                        data = audio.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    sink.feed(data)
            sink.flush()
        except Exception as e:                           # noqa: BLE001
            log.warning("Meeting copilot microphone stopped: %s", e)
            if not self._stop.is_set():
                self.note.emit("Microphone stopped: %s" % native.mic_error(str(e)))

    def _run_system(self, make_sink):
        if native.OS == "windows":                       # soundcard talks COM; every thread must join it first
            import ctypes
            ctypes.windll.ole32.CoInitializeEx(None, 0x0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")          # soundcard warns about discontinuities; harmless here
                import numpy as np
                import soundcard as sc
                speaker = sc.default_speaker()
                loop = sc.get_microphone(id=str(speaker.name), include_loopback=True)
                sink = make_sink(THEM, RATE)
                frames = int(RATE * BLOCK_SEC)
                with loop.recorder(samplerate=RATE, channels=1, blocksize=frames) as r:
                    log.info("Meeting copilot hearing system audio from %s", speaker.name)
                    while not self._stop.is_set():
                        block = r.record(numframes=frames)
                        if block is None or not len(block):
                            time.sleep(0.02)
                            continue
                        mono = block.mean(axis=1) if block.ndim > 1 else block
                        pcm = (np.clip(mono, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                        sink.feed(pcm)
                sink.flush()
        except Exception as e:                           # noqa: BLE001
            log.warning("Meeting copilot system audio stopped: %s", e)
            if not self._stop.is_set():
                self.note.emit("Couldn't hear the computer's sound (%s). Transcribing your microphone only." % e)
