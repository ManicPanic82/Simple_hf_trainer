"""MainWindow closeEvent shutdown path: bounded polling, feedback and escalation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMainWindow

from ui.main_window import SHUTDOWN_TIMEOUT_MS, MainWindow


def _tab(*, running: bool = False) -> MagicMock:
    """A tab double exposing the worker surface MainWindow relies on."""
    tab = MagicMock()
    tab.worker_running = running
    tab.worker = MagicMock()
    return tab


@pytest.fixture
def window(context, qapp):
    """A real MainWindow whose tabs, banner and close() are deterministic doubles."""
    from app.application import create_window

    win = create_window(context)
    win.context = MagicMock()
    win.banner = MagicMock()
    win.close = MagicMock()
    win.training_tab = _tab()
    win.heretic_tab = _tab()
    yield win
    for name in ("_shutdown_timer", "_shutdown_timeout_timer"):
        timer = getattr(win, name, None)
        if timer is not None:
            timer.stop()
    win.deleteLater()
    QApplication.processEvents()


def _close_event() -> MagicMock:
    return MagicMock(spec=QCloseEvent)


def test_initial_discovery_is_not_scheduled_during_construction(
    window: MainWindow, wait_for
) -> None:
    with (
        patch.object(window.models_tab, "scan_library") as model_scan,
        patch.object(window.datasets_tab, "scan_datasets") as dataset_scan,
    ):
        assert not window._models_scan_timer.isActive()
        assert not window._datasets_scan_timer.isActive()

        # Process the event loop beyond the former singleShot delay. Construction
        # alone must not launch a scan before MainWindow.start() is called.
        assert wait_for(lambda: False, timeout_ms=150) is False
        model_scan.assert_not_called()
        dataset_scan.assert_not_called()


def test_start_schedules_both_initial_discovery_scans(window: MainWindow) -> None:
    window.start()

    assert window._models_scan_timer.isActive()
    assert window._datasets_scan_timer.isActive()

    window._stop_startup_scan_timers()


def test_clean_close_calls_super_and_accepts(window: MainWindow) -> None:
    event = _close_event()

    def _accept(*_args):
        event.accept()

    with patch.object(QMainWindow, "closeEvent", side_effect=_accept) as super_close:
        window.closeEvent(event)

    super_close.assert_called_once()
    event.accept.assert_called_once_with()
    event.ignore.assert_not_called()
    assert not getattr(window, "_shutdown_pending", False)


def test_active_worker_ignores_close_notices_and_starts_timers(window: MainWindow) -> None:
    window.training_tab.worker_running = True
    event = _close_event()

    window.closeEvent(event)

    event.ignore.assert_called_once_with()
    event.accept.assert_not_called()
    notices = [call.args[0] for call in window.banner.show_notice.call_args_list]
    assert any(n.level == "info" and "Stopping active workers" in n.message for n in notices)
    window.training_tab.request_shutdown.assert_called_once_with()
    window.heretic_tab.request_shutdown.assert_called_once_with()
    assert window._shutdown_timer.isActive()
    assert window._shutdown_timeout_timer.isActive()
    assert window._shutdown_timeout_timer.interval() == SHUTDOWN_TIMEOUT_MS
    window.close.assert_not_called()


def test_repeated_close_attempts_do_not_duplicate_timers(window: MainWindow) -> None:
    window.heretic_tab.worker_running = True

    window.closeEvent(_close_event())
    poll_timer = window._shutdown_timer
    timeout_timer = window._shutdown_timeout_timer
    second_event = _close_event()
    window.closeEvent(second_event)

    assert window._shutdown_timer is poll_timer
    assert window._shutdown_timeout_timer is timeout_timer
    assert window.training_tab.request_shutdown.call_count == 1
    assert window.heretic_tab.request_shutdown.call_count == 1
    second_event.ignore.assert_called_once_with()
    window.close.assert_not_called()


def test_worker_exit_closes_window_and_stops_timers(window: MainWindow) -> None:
    window.training_tab.worker_running = True
    window.closeEvent(_close_event())

    window.training_tab.worker_running = False
    window.heretic_tab.worker_running = False
    window._shutdown_timer.timeout.emit()

    window.close.assert_called_once_with()
    assert not window._shutdown_timer.isActive()
    assert not window._shutdown_timeout_timer.isActive()


def test_shutdown_waits_for_local_discovery_thread(window: MainWindow) -> None:
    scan = MagicMock()
    scan.isRunning.return_value = True
    window.models_tab._scan_worker = scan

    window.closeEvent(_close_event())
    window._shutdown_timeout_timer.timeout.emit()

    window.close.assert_not_called()
    assert window._shutdown_timeout_timer.isActive()
    notices = [call.args[0] for call in window.banner.show_notice.call_args_list]
    assert any("Waiting for local library discovery" in n.message for n in notices)

    scan.isRunning.return_value = False
    window._shutdown_timer.timeout.emit()

    window.close.assert_called_once_with()


def test_timeout_force_stops_workers_and_closes(window: MainWindow) -> None:
    window.training_tab.worker_running = True
    window.heretic_tab.worker_running = True
    window.closeEvent(_close_event())

    window._shutdown_timeout_timer.timeout.emit()

    window.training_tab.worker.stop.assert_called_once_with()
    window.heretic_tab.worker.stop.assert_called_once_with()
    window.close.assert_called_once_with()
    assert not window._shutdown_timer.isActive()
    assert not window._shutdown_timeout_timer.isActive()
    assert window._shutdown_escalated is True
    notices = [call.args[0] for call in window.banner.show_notice.call_args_list]
    assert any(n.level == "warning" and "Forcing" in n.message for n in notices)

    # A repeated timeout must not escalate a second time.
    window._shutdown_timeout_timer.timeout.emit()
    window.training_tab.worker.stop.assert_called_once_with()
    window.heretic_tab.worker.stop.assert_called_once_with()
    window.close.assert_called_once_with()


def test_forced_stop_failure_still_closes_and_bypasses_worker_check(window: MainWindow) -> None:
    window.training_tab.worker_running = True
    window.training_tab.worker.stop.side_effect = RuntimeError("stop failed")
    window.closeEvent(_close_event())

    window._escalate_shutdown()

    window.close.assert_called_once_with()
    assert window._force_close is True

    final_event = _close_event()
    with patch.object(QMainWindow, "closeEvent") as super_close:
        window.closeEvent(final_event)
    super_close.assert_called_once()
    final_event.ignore.assert_not_called()


def test_shutdown_tolerates_tabs_without_worker_attributes(window: MainWindow) -> None:
    bare = MagicMock(spec=["request_shutdown"])
    window.training_tab = bare
    window.heretic_tab = bare

    assert window._workers_running() is False
    window._force_stop_worker(bare)  # no worker/_active_worker attribute: no-op
    window._poll_worker_shutdown()  # no worker attribute: treat as exited and close
    window.close.assert_called_once_with()


def test_shutdown_also_stops_hub_threads(window: MainWindow) -> None:
    window.models_tab = MagicMock()
    window.datasets_tab = MagicMock()
    window.training_tab.worker_running = True

    window.closeEvent(_close_event())

    window.models_tab.request_shutdown.assert_called_once_with()
    window.datasets_tab.request_shutdown.assert_called_once_with()


def test_models_tab_request_shutdown_cancels_download_and_joins_threads() -> None:
    from ui.models_tab import ModelsTab

    tab = ModelsTab.__new__(ModelsTab)
    download = MagicMock()
    search = MagicMock()
    info = MagicMock()
    scan = MagicMock()
    tab._download_worker = download
    tab._search_worker = search
    tab._scan_worker = scan
    tab._info_workers = [info]

    ModelsTab.request_shutdown(tab)

    download.cancel.assert_called_once_with()
    for worker in (download, search, scan, info):
        worker.requestInterruption.assert_called_once_with()
        worker.quit.assert_called_once_with()
        worker.wait.assert_called_once_with(500)


def test_datasets_tab_request_shutdown_joins_threads() -> None:
    from ui.datasets_tab import DatasetsTab

    tab = DatasetsTab.__new__(DatasetsTab)
    search = MagicMock()
    download = MagicMock()
    scan = MagicMock()
    tab._search_worker = search
    tab._download_worker = download
    tab._scan_worker = scan

    DatasetsTab.request_shutdown(tab)

    for worker in (search, download, scan):
        worker.requestInterruption.assert_called_once_with()
        worker.quit.assert_called_once_with()
        worker.wait.assert_called_once_with(500)
