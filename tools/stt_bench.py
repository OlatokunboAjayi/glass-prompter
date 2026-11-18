"""Speech-to-text benchmark: the copilot's Vosk model against faster-whisper sizes.

Windows only (speaks the test sentences with the built-in System.Speech voices).
    python tools/stt_bench.py [--engines vosk,tiny.en,base.en,small.en] [--noise-db 10]
Prints word error rate (lower is better) and real-time factor (seconds of compute per second of audio).
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RATE = 16000
# Original meeting-style lines: questions, numbers, names and jargon, the things the copilot must hear right.
SENTENCES = [
    "So what does pricing look like for a team of twelve people",
    "We shipped the new onboarding flow last Thursday and churn dropped by four percent",
    "Can you walk me through how the integration with Salesforce works",
    "Our budget for the pilot is around fifteen thousand dollars per quarter",
    "I think the main blocker is security review from the IT department",
    "Does the annual plan include priority support and a dedicated account manager",
    "Let's schedule a follow up call for next Tuesday at two thirty",
    "How long does it usually take to migrate about forty gigabytes of data",
    "Tell me about a time you disagreed with your manager and how you handled it",
    "We need the contract signed before the end of the fiscal year in March",
    "Is the data encrypted at rest and in transit",
    "Thanks everyone, I'll send the summary and action items by email",
]


def _norm(s):
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", s.lower().replace("-", " ")).split())


_NUM = {"twelve": "12", "four": "4", "fifteen": "15", "thousand": "000", "forty": "40", "two": "2", "thirty": "30"}


def _canon(s):
    """Numbers are fine either way: Whisper writes '12', Vosk writes 'twelve'."""
    words = _norm(s).replace("percent", " percent").split()
    out = []
    for w in words:
        out.append(_NUM.get(w, w))
    text = " ".join(out)
    text = re.sub(r"\b(\d+) 000\b", r"\g<1>000", text)
    text = re.sub(r"\b(\d+)\s*%", r"\1 percent", text)
    text = re.sub(r"\b2 30\b", "230", text).replace("2:30", "230")
    return text.replace("i'll", "i will")


def wer(ref, hyp):
    import jiwer
    return jiwer.wer(_canon(ref), _canon(hyp) or "<empty>")


def speak_all(folder):
    """One 16 kHz mono WAV per sentence via Windows System.Speech."""
    lines = ["Add-Type -AssemblyName System.Speech",
             "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
             "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
             "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)",
             "$s.Rate = 1"]
    paths = []
    for i, text in enumerate(SENTENCES):
        p = os.path.join(folder, "s%02d.wav" % i)
        paths.append(p)
        lines += ["$s.SetOutputToWaveFile('%s', $f)" % p, "$s.Speak('%s')" % text.replace("'", "''")]
    lines.append("$s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", "; ".join(lines)], check=True)
    return paths


def read_wav(path):
    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def add_noise(audio, snr_db, seed):
    """Pink-ish room noise at the given signal-to-noise ratio, like a laptop mic in a busy room."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(len(audio) + 64).astype(np.float32)
    noise = np.convolve(white, np.ones(8) / 8, mode="same")[:len(audio)]
    p_sig, p_noise = np.mean(audio ** 2) + 1e-12, np.mean(noise ** 2) + 1e-12
    return np.clip(audio + noise * np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10))), -1, 1)


def run_vosk(clips):
    import json
    import vosk
    from glassprompter import listen
    vosk.SetLogLevel(-1)
    model = vosk.Model(listen.model_path(""))
    out = []
    for a in clips:
        rec = vosk.KaldiRecognizer(model, RATE)
        rec.AcceptWaveform((a * 32767).astype(np.int16).tobytes())
        out.append(json.loads(rec.FinalResult()).get("text", ""))
    return out


def run_whisper(size, clips):
    from faster_whisper import WhisperModel
    model = WhisperModel(size, device="cpu", compute_type="int8")
    model.transcribe(np.zeros(RATE, dtype=np.float32))                 # warm up outside the timing
    out = []
    for a in clips:
        segs, _ = model.transcribe(a, language="en", beam_size=1, vad_filter=False, condition_on_previous_text=False)
        out.append(" ".join(s.text.strip() for s in segs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="vosk,tiny.en,base.en,small.en")
    ap.add_argument("--noise-db", type=float, default=10.0)
    ap.add_argument("--show", action="store_true", help="print each transcript")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        clean = [read_wav(p) for p in speak_all(tmp)]
    noisy = [add_noise(a, args.noise_db, i) for i, a in enumerate(clean)]
    seconds = sum(len(a) for a in clean) / RATE
    print("%d sentences, %.0f s of speech, noise at %g dB SNR\n" % (len(clean), seconds, args.noise_db))
    print("%-10s %12s %12s %8s" % ("engine", "WER clean", "WER noisy", "RTF"))
    for eng in args.engines.split(","):
        row = []
        t0 = time.perf_counter()
        for clips in (clean, noisy):
            hyps = run_vosk(clips) if eng == "vosk" else run_whisper(eng, clips)
            row.append(sum(wer(r, h) for r, h in zip(SENTENCES, hyps)) / len(SENTENCES))
            if args.show:
                for r, h in zip(SENTENCES, hyps):
                    print("   ", "OK " if wer(r, h) == 0 else "ERR", h)
        rtf = (time.perf_counter() - t0) / (2 * seconds)
        print("%-10s %11.1f%% %11.1f%% %8.2f" % (eng, row[0] * 100, row[1] * 100, rtf), flush=True)


if __name__ == "__main__":
    main()
