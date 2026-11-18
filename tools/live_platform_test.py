"""Live, real-screen test of a fresh install: glass on every window, a copilot meeting through Whisper, hide-all.

    python tools/live_platform_test.py [out_dir]

Needs a desktop session (not offscreen) and speakers: it plays meeting lines with the Windows voice. Glass is
measured, not eyeballed: a pure red window sits behind each window and the screen pixels inside it are read back.
"""
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["GLASSPROMPTER_HOME"] = tempfile.mkdtemp(prefix="gp_live_")      # a brand-new install
os.environ["GLASSPROMPTER_ALLOW_CAPTURE"] = "1"                               # so the screen grab can see our windows

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QCursor, QGuiApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from glassprompter import app as gpapp, config, meeting  # noqa: E402
from glassprompter.ui import theme  # noqa: E402
from glassprompter.ui.glass import GlassDialog  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), "gp_live_shots")
os.makedirs(out, exist_ok=True)
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("%-44s %s  %s" % (name, "PASS" if ok else "FAIL", detail), flush=True)


app = QApplication(sys.argv[:1])
app.setStyle("Fusion")
app.setStyleSheet(theme.app_qss())


def pump(seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.015)


cfg = config.Config()
cfg.load()
check("fresh install defaults to Glass", cfg.s.window_glass == 0.7, "window_glass=%s" % cfg.s.window_glass)
cfg.s.first_run_done = True
cfg.save()


class Args:
    background = True
    file = None


ctl = gpapp.Controller(app, Args())
screen = QGuiApplication.primaryScreen()
area = screen.availableGeometry()
red = QWidget(None, Qt.WindowType.FramelessWindowHint)
red.setStyleSheet("background: #FF0000;")
red.setGeometry(area)
red.show()
red.raise_()                                            # above other apps; our windows stay on top of it
ctl.prompter.hide_window()                              # it stays on top and would cover the windows being measured
QCursor.setPos(area.left() + 2, area.bottom() - 2)      # mouse away from the windows, so none turns solid on hover
pump(1.0)


def redness(w):
    """Median red-minus-blue over a grid of points inside the window, read from the real screen."""
    g = w.frameGeometry()
    for corner in (area.bottomLeft(), area.bottomRight(), area.topRight(), area.topLeft()):
        if not g.adjusted(-4, -4, 4, 4).contains(corner):           # park the mouse off this window
            QCursor.setPos(corner)
            break
    pump(0.5)                                                       # the hover timer notices and drops to Glass
    img = screen.grabWindow(0).toImage()
    ratio = screen.devicePixelRatio()
    g = w.frameGeometry()
    vals = []
    for fx in (0.15, 0.3, 0.5, 0.7, 0.85):
        for fy in (0.2, 0.4, 0.6, 0.8):
            c = img.pixelColor(int((g.x() + g.width() * fx - area.x() + area.x()) * ratio),
                               int((g.y() + g.height() * fy) * ratio))
            vals.append(c.red() - c.blue())
    vals.sort()
    return vals[len(vals) // 2]


def show_and_measure(label, opener, getter):
    opener()
    pump(1.2)
    w = getter()
    if w is None or not w.isVisible():
        check(label + " opens", False, "no window")
        return None
    w.raise_()
    pump(0.6)
    r = redness(w)
    w.grab().save(os.path.join(out, label.lower().replace(" ", "_") + ".png"))
    check(label + " is see-through (Glass)", r > 40, "red-blue %d" % r)
    return w


def close(w):
    if w is not None:
        w.close()
        pump(0.5)


# ----------------------------------------------------------------- glass on every window
for label, opener, getter in (("Settings", ctl.open_settings, lambda: ctl.settings_win),
                               ("Meeting history", ctl.open_meetings, lambda: ctl.meetings_window)):
    close(show_and_measure(label, opener, getter))

from PySide6.QtCore import QTimer  # noqa: E402
for label, opener in (("Library", ctl.open_library), ("Phone remote", ctl.open_phone), ("About", ctl.open_about)):
    # modal: measure from inside its own event loop, then close it so the next one can open
    QTimer.singleShot(0, lambda lb=label: (show_and_measure(lb, lambda: None, lambda: ctl.dialog),
                                           ctl.dialog.reject() if ctl.dialog else None))
    opener()
    pump(0.4)

ctl.open_copilot()
pump(1.0)
cop = ctl.copilot
show_and_measure("Copilot", lambda: None, lambda: cop)
def inspect_setup():
    setup = next((w for w in QApplication.topLevelWidgets() if type(w).__name__ == "Setup" and w.isVisible()), None)
    if setup is None:
        return check("Copilot context opens", False)
    cop.hide()                                          # the copilot sits right behind its own context window
    pump(0.8)
    check("Copilot context is see-through (Glass)", redness(setup) > 40, "red-blue %d" % redness(setup))
    cop.show()
    texts = [setup.speech.itemText(i) for i in range(setup.speech.count())]
    check("Copilot context offers Whisper engines", any("Whisper Base" in t for t in texts), "%d engines" % len(texts))
    setup.grab().save(os.path.join(out, "copilot_context.png"))
    setup.reject()


QTimer.singleShot(0, inspect_setup)                       # modal, like the dialogs above
cop.open_setup()
pump(0.4)

GlassDialog.set_level(1.0)                                # control: Solid must hide the red completely
pump(0.6)
check("Solid level blocks the background (control)", redness(cop) < 20, "red-blue %d" % redness(cop))
GlassDialog.set_level(0.7)
pump(0.6)
check("Copilot back to Glass", redness(cop) > 40, "red-blue %d" % redness(cop))

# ----------------------------------------------------------------- a live meeting through Whisper
LINES = ["Hi, thanks for joining. So what does pricing look like for a team of twelve people?",
         "And is the data encrypted at rest and in transit?"]
phrases = []                                            # what Smart answers sees: each phrase as it arrives
cop.listener.segment.connect(lambda who, t: phrases.append(t) if who == meeting.THEM else None)
cop.start_meeting()
t0 = time.time()
while not (cop.listener.listening and cop.listener.engine_name) and time.time() - t0 < 300:
    pump(0.3)
pump(2.0)
check("Copilot meeting runs on Whisper", "Whisper" in cop.listener.engine_name, cop.listener.engine_name)
say = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
       + " ".join("$s.Speak('%s'); Start-Sleep -Milliseconds 1500;" % s.replace("'", "''") for s in LINES))
p = subprocess.Popen(["powershell", "-NoProfile", "-Command", say])
while p.poll() is None:
    pump(0.2)
pump(5.0)
them = [s["text"] for s in cop.transcript.segments if s["who"] == meeting.THEM]
heard = " ".join(them)
check("Their words reach the live transcript", "pricing" in heard.lower() and "encrypted" in heard.lower(),
      heard[:110])
check("Both questions are spotted as they arrive", sum(meeting.is_question(t) for t in phrases) >= 2,
      "%d of %d phrases" % (sum(meeting.is_question(t) for t in phrases), len(phrases)))
cop.grab().save(os.path.join(out, "copilot_live_meeting.png"))
cop.toggle_meeting()
pump(2.0)
check("Meeting stops cleanly", not cop.listener.listening)

# ----------------------------------------------------------------- hide everything
ctl.prompter.show_window()
ctl.open_settings()
pump(1.0)
mine = [w for w in (ctl.prompter, cop, ctl.settings_win) if w is not None and w.isVisible()]
ctl.toggle_everything()
pump(0.6)
check("Hide-all hides every window", not any(w.isVisible() for w in mine), "%d windows" % len(mine))
ctl.toggle_everything()
pump(0.6)
check("Hide-all brings them all back", all(w.isVisible() for w in mine))

red.close()
print("\nRESULT: %s (%d/%d)   screenshots: %s" % ("OK" if all(results) else "FAIL", sum(results), len(results), out))
os._exit(0 if all(results) else 1)
