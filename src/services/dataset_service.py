"""Dataset discovery, local inspection, preview, validation, and training preparation."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.credentials import CredentialStore
from services.storage_service import StorageService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalDatasetRecord:
    """A dataset located inside datasets_dir."""

    path: Path
    name: str
    format: str  # "jsonl" | "json" | "parquet" | "csv" | "unknown"
    row_count: int | None
    size_mb: float
    is_prepared: bool
    columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HubDatasetRecord:
    """A dataset search result from Hugging Face Hub."""

    repo_id: str
    author: str
    name: str
    downloads: int
    likes: int
    description: str = ""


@dataclass(frozen=True)
class DatasetPreview:
    """Preview of dataset sample rows, schema, and validation status."""

    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int | None
    format: str
    is_valid_for_training: bool
    validation_message: str


def inspect_dataset_file(file_path: Path) -> LocalDatasetRecord | None:
    """Inspect a single file or directory to determine dataset properties."""
    if not file_path.exists():
        return None

    if file_path.is_file():
        suffix = file_path.suffix.lower()
        if suffix not in (".jsonl", ".json", ".csv", ".parquet", ".arrow"):
            return None

        size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
        fmt = suffix.lstrip(".")
        columns: list[str] = []
        row_count: int | None = None

        if fmt == "jsonl":
            count = 0
            with open(file_path, encoding="utf-8", errors="replace") as f:
                for idx, line in enumerate(f):
                    count += 1
                    if idx == 0:
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                columns = list(obj.keys())
                        except Exception:
                            pass
            row_count = count

        elif fmt == "csv":
            count = 0
            with open(file_path, encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    columns = [col.strip() for col in header]
                for _ in reader:
                    count += 1
            row_count = count

        elif fmt == "json":
            try:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        row_count = len(data)
                        if data and isinstance(data[0], dict):
                            columns = list(data[0].keys())
            except Exception:
                pass

        elif fmt == "parquet":
            try:
                import pyarrow.parquet as pq

                meta = pq.read_metadata(file_path)
                row_count = meta.num_rows
                columns = meta.schema.names
            except Exception:
                pass

        is_prepared = file_path.name.startswith("prepared_") or (
            "text" in columns or ("instruction" in columns and "output" in columns)
        )

        return LocalDatasetRecord(
            path=file_path,
            name=file_path.name,
            format=fmt,
            row_count=row_count,
            size_mb=size_mb,
            is_prepared=is_prepared,
            columns=columns,
        )

    return None


class DatasetService:
    """Manages dataset discovery, previewing, and preparation for training."""

    def __init__(
        self,
        storage: StorageService,
        credentials: CredentialStore | None = None,
    ) -> None:
        self._storage = storage
        self._credentials = credentials

    def _get_token(self) -> str | None:
        if self._credentials and self._credentials.available:
            return self._credentials.get_token()
        return None

    def scan_local_datasets(self) -> list[LocalDatasetRecord]:
        """Scan datasets_dir for local datasets."""
        validation = self._storage.validate("datasets_dir", probe_writable=False)
        if not validation.exists or not validation.path.is_dir():
            return []

        records: list[LocalDatasetRecord] = []
        root = validation.path

        try:
            for item in root.iterdir():
                rec = inspect_dataset_file(item)
                if rec is not None:
                    records.append(rec)
        except OSError as error:
            logger.warning("Error reading datasets directory %s: %s", root, error)

        return sorted(records, key=lambda r: (not r.is_prepared, r.name.lower()))

    def preview_dataset(self, path: Path | str, num_rows: int = 10) -> DatasetPreview:
        """Load the first few rows of a dataset and validate its structure."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        suffix = file_path.suffix.lower()
        rows: list[dict[str, Any]] = []
        columns: list[str] = []
        total_rows: int | None = None
        fmt = suffix.lstrip(".")

        if suffix == ".jsonl":
            count = 0
            with open(file_path, encoding="utf-8", errors="replace") as f:
                for idx, line in enumerate(f):
                    count += 1
                    if idx < num_rows:
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                if not columns:
                                    columns = list(obj.keys())
                                rows.append(obj)
                        except Exception:
                            pass
            total_rows = count

        elif suffix == ".json":
            with open(file_path, encoding="utf-8", errors="replace") as f:
                data = json.load(f)
                if isinstance(data, list):
                    total_rows = len(data)
                    for item in data[:num_rows]:
                        if isinstance(item, dict):
                            if not columns:
                                columns = list(item.keys())
                            rows.append(item)

        elif suffix == ".csv":
            count = 0
            with open(file_path, encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    columns = list(reader.fieldnames)
                for idx, row in enumerate(reader):
                    count += 1
                    if idx < num_rows:
                        rows.append(dict(row))
            total_rows = count

        elif suffix == ".parquet":
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(file_path)
                columns = table.column_names
                total_rows = table.num_rows
                slice_table = table.slice(0, num_rows)
                rows = slice_table.to_pylist()
            except Exception as error:
                return DatasetPreview(
                    columns=[],
                    rows=[],
                    total_rows=None,
                    format=fmt,
                    is_valid_for_training=False,
                    validation_message=f"Failed to read Parquet: {error}",
                )

        col_set = set(columns)
        has_text = "text" in col_set
        has_alpaca = "instruction" in col_set and "output" in col_set
        has_prompt_response = ("prompt" in col_set and "response" in col_set) or (
            "prompt" in col_set and "completion" in col_set
        )

        if has_text or has_alpaca or has_prompt_response:
            valid = True
            msg = "Ready for fine-tuning. Training columns detected."
        else:
            valid = False
            msg = (
                f"Needs preparation: expected ('text') or ('instruction', 'output') columns. "
                f"Found: {', '.join(columns[:5]) if columns else 'none'}."
            )

        return DatasetPreview(
            columns=columns,
            rows=rows,
            total_rows=total_rows,
            format=fmt,
            is_valid_for_training=valid,
            validation_message=msg,
        )

    def prepare_dataset(
        self,
        source_path: Path,
        output_path: Path | None = None,
        instruction_col: str = "instruction",
        input_col: str = "",
        output_col: str = "output",
        text_template: str = "alpaca",
    ) -> Path:
        """Transform a dataset into a standard training-ready JSONL file."""
        if not source_path.is_file():
            raise FileNotFoundError(f"Source file does not exist: {source_path}")

        if output_path is None:
            datasets_dir = self._storage.path_for("datasets_dir")
            stem = source_path.stem
            output_path = datasets_dir / f"prepared_{stem}.jsonl"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        preview = self.preview_dataset(source_path, num_rows=1)
        fmt = preview.format

        def format_alpaca(inst: str, inp: str, out: str) -> str:
            if inp.strip():
                return (
                    f"### Instruction:\n{inst.strip()}\n\n"
                    f"### Input:\n{inp.strip()}\n\n"
                    f"### Response:\n{out.strip()}"
                )
            return f"### Instruction:\n{inst.strip()}\n\n### Response:\n{out.strip()}"

        rows_written = 0
        with open(output_path, "w", encoding="utf-8") as out_f:
            if fmt == "jsonl":
                with open(source_path, encoding="utf-8", errors="replace") as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if not isinstance(obj, dict):
                                continue
                            inst = str(obj.get(instruction_col, "") or "")
                            inp = str(obj.get(input_col, "") or "") if input_col else ""
                            resp = str(obj.get(output_col, "") or "")
                            formatted_text = format_alpaca(inst, inp, resp)
                            record = {
                                "instruction": inst,
                                "input": inp,
                                "output": resp,
                                "text": formatted_text,
                            }
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            rows_written += 1
                        except json.JSONDecodeError:
                            continue

            elif fmt == "csv":
                with open(source_path, encoding="utf-8", errors="replace", newline="") as in_f:
                    reader = csv.DictReader(in_f)
                    for row in reader:
                        inst = str(row.get(instruction_col, "") or "")
                        inp = str(row.get(input_col, "") or "") if input_col else ""
                        resp = str(row.get(output_col, "") or "")
                        formatted_text = format_alpaca(inst, inp, resp)
                        record = {
                            "instruction": inst,
                            "input": inp,
                            "output": resp,
                            "text": formatted_text,
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        rows_written += 1

            elif fmt == "parquet":
                import pyarrow.parquet as pq

                table = pq.read_table(source_path)
                pylist = table.to_pylist()
                for row in pylist:
                    inst = str(row.get(instruction_col, "") or "")
                    inp = str(row.get(input_col, "") or "") if input_col else ""
                    resp = str(row.get(output_col, "") or "")
                    formatted_text = format_alpaca(inst, inp, resp)
                    record = {
                        "instruction": inst,
                        "input": inp,
                        "output": resp,
                        "text": formatted_text,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rows_written += 1

            elif fmt == "json":
                with open(source_path, encoding="utf-8", errors="replace") as in_f:
                    data = json.load(in_f)
                    if isinstance(data, list):
                        for row in data:
                            if not isinstance(row, dict):
                                continue
                            inst = str(row.get(instruction_col, "") or "")
                            inp = str(row.get(input_col, "") or "") if input_col else ""
                            resp = str(row.get(output_col, "") or "")
                            formatted_text = format_alpaca(inst, inp, resp)
                            record = {
                                "instruction": inst,
                                "input": inp,
                                "output": resp,
                                "text": formatted_text,
                            }
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            rows_written += 1

        logger.info("Prepared %d rows written to %s", rows_written, output_path)
        return output_path

    def search_hub(self, query: str, limit: int = 20) -> list[HubDatasetRecord]:
        """Search Hugging Face Hub datasets."""
        from huggingface_hub import HfApi

        token = self._get_token()
        api = HfApi(token=token)
        try:
            results = list(
                api.list_datasets(
                    search=query.strip(),
                    limit=limit,
                    full=False,
                )
            )
            records: list[HubDatasetRecord] = []
            for item in results:
                ds_id = getattr(item, "id", "") or getattr(item, "datasetId", "")
                if not ds_id:
                    continue
                parts = ds_id.split("/")
                author = parts[0] if len(parts) > 1 else "community"
                name = parts[-1]
                records.append(
                    HubDatasetRecord(
                        repo_id=ds_id,
                        author=author,
                        name=name,
                        downloads=int(getattr(item, "downloads", 0) or 0),
                        likes=int(getattr(item, "likes", 0) or 0),
                        description=str(getattr(item, "description", "") or ""),
                    )
                )
            return records
        except Exception as error:
            logger.error("Hub dataset search failed: %s", error)
            raise
