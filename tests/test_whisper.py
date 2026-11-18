import numpy as np
import pytest

from glassprompter import listen, meeting, whisper_stt

RATE = whisper_stt.RATE


def _tone(seconds, amp=0.2):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _quiet(seconds):
    return (np.random.default_rng(0).standard_normal(int(seconds * RATE)) * 0.001).astype(np.float32)


def test_segmenter_cuts_phrases_at_pauses_and_drops_blips():
    seg = whisper_stt.Segmenter()
    assert seg.feed(_quiet(1.0)) is None and not seg.speaking          # learns the room, nothing to say
    assert seg.feed(_tone(1.5)) is None and seg.speaking
    phrase = seg.feed(_quiet(1.0))                                     # a 0.7 s pause ends the phrase
    assert phrase is not None and 1.5 <= len(phrase) / RATE <= 2.6 and not seg.speaking
    seg.feed(_tone(0.1))                                               # a click is not speech
    assert seg.feed(_quiet(1.0)) is None


def test_segmenter_cuts_a_long_monologue():
    seg = whisper_stt.Segmenter()
    seg.feed(_quiet(0.5))
    phrase = seg.feed(_tone(16.0))
    assert phrase is not None and len(phrase) / RATE <= whisper_stt.Segmenter.MAX_PHRASE + 0.5


def test_audio_is_resampled_to_16k():
    pcm = (np.zeros(48000, dtype="<i2")).tobytes()
    assert len(whisper_stt.to_16k(pcm, 48000)) == 16000 and len(whisper_stt.to_16k(pcm, 16000)) == 48000


def test_engine_choice(monkeypatch):
    monkeypatch.setattr(whisper_stt, "faster_whisper", object())
    assert listen.engine("") == ("whisper", whisper_stt.DEFAULT_SIZE)
    assert listen.engine("whisper:small.en") == ("whisper", "small.en")
    assert listen.engine("whisper:nonsense")[0] == "vosk"
    assert listen.engine("vosk-model-small-en-us-0.15")[0] == "vosk"
    monkeypatch.setattr(whisper_stt, "faster_whisper", None)
    assert listen.engine("")[0] == "vosk" and listen.engine("whisper:base.en")[0] == "vosk"


def test_answer_language_reaches_the_prompt():
    ctx = meeting.build_context("sales", [], None, language="Spanish")
    assert "Always write your answers in Spanish" in ctx
    assert "Always write" not in meeting.build_context("sales", [], None)


def test_decoder_turns_queued_phrases_into_segments():
    got = []

    class FakeModel:
        def transcribe(self, audio, **kw):
            seg = type("S", (), {"text": " Hello there.", "no_speech_prob": 0.1, "avg_logprob": -0.2})
            return [seg], None
    dec = whisper_stt.Decoder(FakeModel(), "en", "Acme", lambda who, t: got.append((who, t)), lambda *_: None)
    sink = dec.sink(meeting.THEM, RATE)
    sink.feed((np.concatenate([_quiet(0.5), _tone(1.0), _quiet(1.0)]) * 32767).astype("<i2").tobytes())
    who, audio = dec.finals.get_nowait()
    assert who == meeting.THEM and dec.prompt == "Acme"
    assert dec.transcribe(audio) == "Hello there."


@pytest.mark.skipif(not whisper_stt.available() or not whisper_stt.is_downloaded("tiny.en"),
                    reason="needs faster-whisper and the tiny.en model")
def test_real_whisper_hears_silence_as_nothing():
    dec = whisper_stt.Decoder(whisper_stt.load("tiny.en"), "en", "", lambda *_: None, lambda *_: None)
    assert dec.transcribe(_quiet(2.0)) == ""


def test_hide_everything_hides_and_restores_exactly_what_was_open():
    from PySide6.QtWidgets import QApplication, QWidget
    QApplication.instance() or QApplication([])
    from glassprompter.app import Controller

    class P(QWidget):
        def show_window(self):
            self.show()

        def hide_window(self):
            self.hide()
    c = Controller.__new__(Controller)
    c.prompter, other, closed = P(), QWidget(), QWidget()
    c.prompter.show()
    other.show()
    c.toggle_everything()
    assert not c.prompter.isVisible() and not other.isVisible()
    c.toggle_everything()
    assert c.prompter.isVisible() and other.isVisible() and not closed.isVisible()
    for w in (c.prompter, other):
        w.hide()
    c.toggle_everything()                                              # nothing open: the prompter comes up
    assert c.prompter.isVisible()
    c.prompter.hide()
