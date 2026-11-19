"""Every window must fit, uncut, on the screens people actually have, including small laptops at 125-150% scaling."""
import os
import sys
import tempfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GLASSPROMPTER_ALLOW_CAPTURE", "1")
pytest.importorskip("PySide6")

from PySide6.QtCore import QRect  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

# usable area in logical points (what Qt sees after display scaling)
SCREENS = {
    "1280x800 laptop at 150%, taskbar hidden": QRect(0, 0, 1280, 752),
    "1366x768 laptop at 125%": QRect(0, 0, 1093, 574),
    "1920x1080 at 150%": QRect(0, 0, 1280, 680),
    "1024x600 netbook": QRect(0, 0, 1024, 560),
}


@pytest.fixture(scope="module")
def ctl():
    os.environ["GLASSPROMPTER_HOME"] = tempfile.mkdtemp(prefix="gp_size_")
    from glassprompter import app as gpapp, config
    app = QApplication.instance() or QApplication([])
    c = config.Config()
    c.load()
    c.s.first_run_done = True
    c.save()

    class Args:
        background = True
        file = None
    return gpapp.Controller(app, Args())


def _windows(ctl):
    from glassprompter.ui.assistant import make_assistant
    from glassprompter.ui.copilot import SetupDialog
    from glassprompter.ui.copilot_meetings import Meetings
    from glassprompter.ui.dialogs import AboutDialog, LibraryDialog, PhoneDialog, SettingsDialog
    cop = ctl.ensure_copilot()
    yield "Settings", SettingsDialog(None, ctl)
    yield "Library", LibraryDialog(None, ctl.store, None, lambda sid: None)
    yield "Phone remote", PhoneDialog(None, ctl)
    yield "About", AboutDialog(None, lambda: None, lambda kind="bug": None)
    yield "Meeting history", Meetings(ctl)
    yield "Copilot", cop
    yield "Copilot context", SetupDialog(cop, ctl)
    yield "AI assistant", make_assistant(ctl)


NETBOOK_ON_MAC = pytest.mark.skipif(sys.platform == "darwin", reason="no Mac has a 600-pixel-tall screen; the smallest "
                                    "(11-inch MacBook Air) leaves ~670 points, covered by the 1366x768 case")


@pytest.mark.parametrize("screen", [pytest.param(s, marks=NETBOOK_ON_MAC) if "netbook" in s else s for s in SCREENS])
def test_every_window_fits_uncut(ctl, monkeypatch, screen):
    from glassprompter.ui import glass
    area = SCREENS[screen]
    monkeypatch.setattr(glass, "usable_rect", lambda _s: QRect(area))
    problems = []
    for name, w in _windows(ctl):
        ctl.cfg.s.window_geometry = {}                      # a first open, nothing remembered
        w._placed = False                                   # place it afresh on this screen
        w.show()
        QApplication.processEvents()
        g = w.frameGeometry()
        if not area.contains(g):
            problems.append("%s runs off the screen: %dx%d at (%d,%d)" % (name, g.width(), g.height(), g.x(), g.y()))
        need = w.layout().totalMinimumSize() if w.layout() else w.minimumSizeHint()
        if w.width() < need.width() or w.height() < need.height():
            problems.append("%s cuts its content: %dx%d but needs %dx%d"
                            % (name, w.width(), w.height(), need.width(), need.height()))
        if w.resizable and (g.height() > max(int(area.height() * glass.FIT_H), need.height()) + 2
                            or g.width() > max(int(area.width() * glass.FIT_W), need.width()) + 2):
            problems.append("%s opens too big: %dx%d on a %dx%d screen"
                            % (name, g.width(), g.height(), area.width(), area.height()))
        w.hide()
    assert not problems, "\n".join(problems)
