"""Live test of the meeting copilot's listener: plays meeting lines through the speakers (Windows voice) and checks
what the listener transcribes from the system audio ("Them"). Run once per engine:
    python tools/live_copilot_check.py whisper:base.en
    python tools/live_copilot_check.py vosk-model-small-en-us-0.15
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PySide6.QtCore import QCoreApplication  # noqa: E402

from glassprompter import meeting  # noqa: E402
from glassprompter.listen import MeetingListener  # noqa: E402
from tools.stt_bench import wer  # noqa: E402

LINES = ["So what does pricing look like for a team of twelve people?",
         "Can you walk me through how the integration with Salesforce works?",
         "Is the data encrypted at rest and in transit?",
         "Let's schedule a follow up call for next Tuesday."]


def main():
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    app = QCoreApplication([])
    ls = MeetingListener()
    heard, notes, state = [], [], {}
    ls.segment.connect(lambda who, t: heard.append((who, t)))
    ls.note.connect(notes.append)
    ls.failed.connect(lambda m: notes.append("FAILED " + m))
    ls.status.connect(lambda s: state.__setitem__("s", s))
    ls.start(mic_device="", system_audio=True, model_choice=choice, vocabulary="Salesforce")

    def pump(seconds):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.02)
    t0 = time.time()
    while state.get("s") != "listening" and time.time() - t0 < 600:
        pump(0.2)
    print("engine:", ls.engine_name, "| ready in %.1f s" % (time.time() - t0), "| notes:", notes)
    pump(1.5)
    script = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              + " ".join("$s.Speak('%s'); Start-Sleep -Milliseconds 1200;" % l.replace("'", "''") for l in LINES))
    p = subprocess.Popen(["powershell", "-NoProfile", "-Command", script])
    while p.poll() is None:
        pump(0.2)
    pump(4.0)
    ls.stop()
    pump(2.0)
    them = " ".join(t for who, t in heard if who == meeting.THEM)
    print("THEM heard:", them or "(nothing)")
    print("YOU (mic) heard:", " ".join(t for who, t in heard if who == meeting.ME) or "(nothing)")
    print("questions spotted:", sum(meeting.is_question(t) for who, t in heard if who == meeting.THEM))
    print("WER vs what was played: %.1f%%" % (100 * wer(" ".join(LINES), them)))


if __name__ == "__main__":
    main()
