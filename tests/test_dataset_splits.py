"""Dataset split/file selection in the Datasets tab (no network, mocked service)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from app.state import AppState
from services.dataset_service import DatasetPreview, DatasetService, LocalDatasetRecord
from services.settings_service import AppSettings, PathSettings
from services.storage_service import StorageService
from ui.datasets_tab import DatasetsTab


def _settings(tmp_path: Path) -> AppSettings:
    datasets = tmp_path / "datasets"
    folders = {
        "models_dir": tmp_path / "models",
        "datasets_dir": datasets,
        "outputs_dir": tmp_path / "outputs",
        "hf_cache_dir": tmp_path / "cache",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return AppSettings(paths=PathSettings(**folders))


def _record(path: Path, **overrides: object) -> LocalDatasetRecord:
    values = {
        "path": path,
        "name": path.name,
        "format": "jsonl" if path.is_file() else "dir",
        "row_count": 2,
        "size_mb": 0.01,
        "is_prepared": False,
        "columns": ["instruction", "output"],
    }
    values.update(overrides)
    return LocalDatasetRecord(**values)  # type: ignore[arg-type]


def _service(rec: LocalDatasetRecord, *, valid: bool = True) -> MagicMock:
    service = MagicMock(spec=DatasetService)
    service.scan_local_datasets.return_value = [rec]

    def preview(path: Path | str, num_rows: int = 10) -> DatasetPreview:
        return DatasetPreview(
            columns=["instruction", "output"],
            rows=[{"instruction": "question", "output": "answer"}],
            total_rows=1,
            format=Path(path).suffix.lstrip("."),
            is_valid_for_training=valid,
            validation_message="Ready" if valid else "Needs columns",
        )

    service.preview_dataset.side_effect = preview
    service.prepare_dataset.return_value = rec.path.parent / "prepared.jsonl"
    return service


def _tab(tmp_path: Path, rec: LocalDatasetRecord, *, valid: bool = True) -> DatasetsTab:
    service = _service(rec, valid=valid)
    tab = DatasetsTab(
        AppState(),
        StorageService(_settings(tmp_path)),
        service,
        scan_on_init=False,
    )
    # Populate the table directly: these tests cover split selection, not
    # the asynchronous local-dataset discovery worker.
    tab._on_local_scan_results([rec])
    tab.local_table.selectRow(0)
    return tab

def _two_split_dir(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "datasets" / "org--data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train = data_dir / "train-00000-of-00001.jsonl"
    test = data_dir / "test-00000-of-00001.jsonl"
    train.write_text('{"instruction": "q1", "output": "a1"}\n', encoding="utf-8")
    test.write_text('{"instruction": "q2", "output": "a2"}\n', encoding="utf-8")
    return data_dir, train, test


def _split_index(tab: DatasetsTab, path: Path) -> int:
    return next(
        index for index in range(tab.split_combo.count()) if tab.split_combo.itemData(index) == path
    )


def test_split_combo_lists_real_data_files(qapp, tmp_path) -> None:
    _, train, test = _two_split_dir(tmp_path)
    tab = _tab(tmp_path, _record(train))

    names = [tab.split_combo.itemText(i) for i in range(tab.split_combo.count())]
    assert names == sorted([train.name, test.name], key=str.lower)
    assert tab.split_combo.isEnabled() is True
    # The selected file is the initial split.
    assert tab._selected_split_path == train
    tab._service.preview_dataset.assert_called_with(train, num_rows=10)


def test_selecting_split_repreviews_and_actions_use_selected_file(qapp, tmp_path) -> None:
    _, train, test = _two_split_dir(tmp_path)
    tab = _tab(tmp_path, _record(train))

    tab._service.preview_dataset.reset_mock()
    tab.split_combo.setCurrentIndex(_split_index(tab, test))

    tab._service.preview_dataset.assert_called_once_with(test, num_rows=10)
    assert tab._selected_split_path == test
    assert tab.val_pill.text() == "Ready"
    assert tab.use_in_training_btn.isEnabled() is True

    tab.inst_combo.setCurrentText("instruction")
    tab.inp_combo.setCurrentText("")
    tab.out_combo.setCurrentText("output")
    tab.prepare_button.click()
    tab._service.prepare_dataset.assert_called_once_with(
        test,
        instruction_col="instruction",
        input_col="",
        output_col="output",
    )

    emitted: list[str] = []
    tab.use_dataset_in_training.connect(emitted.append)
    tab.use_in_training_btn.click()
    assert emitted == [str(test)]


def test_invalid_split_updates_readiness_pill(qapp, tmp_path) -> None:
    _, train, test = _two_split_dir(tmp_path)
    service = _service(_record(train))
    original_preview = service.preview_dataset.side_effect

    def preview(path: Path | str, num_rows: int = 10) -> DatasetPreview:
        result: DatasetPreview = original_preview(path, num_rows=num_rows)
        if Path(path) == test:
            return DatasetPreview(
                columns=["id", "value"],
                rows=[{"id": 1, "value": "x"}],
                total_rows=1,
                format="jsonl",
                is_valid_for_training=False,
                validation_message="Needs preparation",
            )
        return result

    service.preview_dataset.side_effect = preview
    tab = DatasetsTab(AppState(), StorageService(_settings(tmp_path)), service)
    tab.local_table.selectRow(0)
    tab.split_combo.setCurrentIndex(_split_index(tab, test))

    assert tab.val_pill.text() == "Needs columns"
    assert tab.use_in_training_btn.isEnabled() is False


def test_single_data_file_disables_split_combo(qapp, tmp_path) -> None:
    data_dir = tmp_path / "datasets" / "single"
    data_dir.mkdir(parents=True, exist_ok=True)
    only = data_dir / "train.jsonl"
    only.write_text('{"instruction": "q", "output": "a"}\n', encoding="utf-8")

    tab = _tab(tmp_path, _record(only))

    assert tab.split_combo.count() == 1
    assert tab.split_combo.isEnabled() is False
    assert tab.split_combo.itemText(0) == only.name
    tab._service.preview_dataset.assert_called_with(only, num_rows=10)


def test_directory_record_lists_only_real_data_files(qapp, tmp_path) -> None:
    data_dir, train, test = _two_split_dir(tmp_path)
    (data_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
    (data_dir / "README.md").write_text("ignore me too", encoding="utf-8")

    tab = _tab(tmp_path, _record(data_dir, name="org--data", format="dir"))

    names = [tab.split_combo.itemText(i) for i in range(tab.split_combo.count())]
    assert names == sorted([train.name, test.name], key=str.lower)
    assert all(name in names for name in (train.name, test.name))
    assert "notes.txt" not in names
