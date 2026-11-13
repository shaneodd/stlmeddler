"""Headless tests for the busy dialog helper (Qt glue, offscreen platform).

``run_busy`` must deliver results and exceptions back on the calling thread,
keep its re-entrancy guard honest, and actually run work off the GUI thread.
"""
import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from src.gui.busy import run_busy  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_returns_result(qapp):
    assert run_busy(None, "Working\u2026", lambda: 42) == 42


def test_exception_reraised_on_caller_thread(qapp):
    def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        run_busy(None, "Working\u2026", boom)


def test_callable_runs_off_main_thread(qapp):
    main_ident = threading.get_ident()

    def where() -> bool:
        return threading.get_ident() != main_ident

    assert run_busy(None, "Working\u2026", where) is True


def test_reentrancy_guard_refuses_nested_call(qapp):
    def nested():
        # While a busy operation is active, any second call must be refused.
        try:
            run_busy(None, "Nested", lambda: None)
        except RuntimeError as exc:
            return str(exc)
        return ""

    message = run_busy(None, "Outer\u2026", nested)
    assert "already running" in message


def test_guard_clears_after_completion(qapp):
    assert run_busy(None, "First\u2026", lambda: 1) == 1
    assert run_busy(None, "Second\u2026", lambda: 2) == 2
