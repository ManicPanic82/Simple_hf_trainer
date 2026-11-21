"""Model discovery, local library scanning, and compatibility checks."""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.credentials import CredentialStore
from services.storage_service import StorageService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalModelRecord:
    """A model found on disk in models_dir or outputs_dir."""

    path: Path
    name: str
    kind: str  # "base" | "adapter" | "abliterated"
    format: str  # "safetensors" | "gguf" | "pytorch" | "unknown"
    architecture: str
    parameter_count: int | None
    size_mb: float
    vram_status: str  # "ok" | "warn" | "error"
    status_detail: str


@dataclass(frozen=True)
class HubModelRecord:
    """A model search result from Hugging Face Hub."""

    repo_id: str
    author: str
    name: str
    downloads: int
    likes: int
    pipeline_tag: str
    private: bool = False


class DownloadCancelled(Exception):
    """A download was stopped between files at the caller's request.

    This is cooperative cancellation only: a file already handed to
    ``hf_hub_download`` is allowed to finish. Files completed before the
    cancellation point are retained on disk.
    """

    def __init__(self, destination: Path, downloaded: int, total: int) -> None:
        self.destination = destination
        self.downloaded = downloaded
        self.total = total
        super().__init__(
            f"Download cancelled after {downloaded}/{total} files; "
            f"partial files were kept in {destination}. "
            "Cancellation is checked between files, not mid-file."
        )


def _is_cancelled(cancel: threading.Event | Callable[[], bool] | None) -> bool:
    """Check a cancellation Event or callable without assuming either type."""
    if cancel is None:
        return False
    if isinstance(cancel, threading.Event):
        return cancel.is_set()
    return bool(cancel())


def estimate_vram_compatibility(
    parameter_count: int | None,
    size_mb: float,
    model_format: str,
) -> tuple[str, str]:
    """Return a conservative 6 GB fit indicator, never a VRAM guarantee.

    Parameter-count tiers are rough 4-bit weight estimates. File size alone
    cannot establish runtime VRAM: quantization, context length, activations,
    offloaded layers, kernels, and other GPU processes all change the result.
    """
    caveat = "Estimate only; actual VRAM also depends on quantization, context length, activations, and GPU headroom."
    if parameter_count is not None:
        if parameter_count <= 1_600_000_000:
            status, estimate = "ok", "small model; likely practical for short-context 4-bit use"
        elif parameter_count <= 3_500_000_000:
            status, estimate = "warn", "may fit with conservative 4-bit settings and short context"
        else:
            status, estimate = (
                "error",
                "high risk of exceeding 6 GB during training or full-GPU loading",
            )
        return status, f"~{parameter_count / 1e9:.1f}B parameters: {estimate}. {caveat}"

    if model_format == "gguf":
        return "warn", (
            f"GGUF file is {size_mb:.0f} MB on disk; file size is not runtime VRAM usage. "
            f"Layer offload and context settings determine GPU memory. {caveat}"
        )

    if size_mb > 0:
        return "warn", (
            f"Model files total {size_mb:.0f} MB on disk; disk footprint cannot predict VRAM. "
            f"Inspect model config or run a small load test. {caveat}"
        )

    return "warn", f"Model size is unknown; no reliable VRAM estimate is available. {caveat}"


def inspect_model_directory(directory: Path) -> LocalModelRecord | None:
    """Inspect a directory to see if it contains a model or adapter."""
    if not directory.is_dir():
        return None

    adapter_config_path = directory / "adapter_config.json"
    config_path = directory / "config.json"
    gguf_files = list(directory.glob("*.gguf"))
    safetensor_files = list(directory.glob("*.safetensors"))
    bin_files = list(directory.glob("*.bin"))

    total_bytes = 0
    for child in directory.rglob("*"):
        if child.is_file():
            with contextlib.suppress(OSError):
                total_bytes += child.stat().st_size
    size_mb = round(total_bytes / (1024 * 1024), 1)

    # 1. Check if it's a LoRA adapter
    if adapter_config_path.is_file():
        arch = "LoRA Adapter"
        try:
            with open(adapter_config_path, encoding="utf-8") as f:
                adapter_cfg = json.load(f)
                base_name = adapter_cfg.get("base_model_name_or_path", "")
                if base_name:
                    arch = f"LoRA Adapter ({Path(base_name).name})"
        except Exception:
            pass
        return LocalModelRecord(
            path=directory,
            name=directory.name,
            kind="adapter",
            format="safetensors" if safetensor_files else "pytorch",
            architecture=arch,
            parameter_count=None,
            size_mb=size_mb,
            vram_status="warn",
            status_detail=(
                f"LoRA adapter ({size_mb:.1f} MB); base-model and training settings "
                "determine VRAM requirements."
            ),
        )

    # 2. Check if it's a full model with config.json
    if config_path.is_file():
        arch = "Unknown"
        model_type = ""
        param_count: int | None = None
        kind = "base"
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
                arch = (
                    (cfg.get("architectures") or [cfg.get("model_type", "Unknown")])[0]
                    if isinstance(cfg.get("architectures"), list)
                    else str(cfg.get("model_type", "Unknown"))
                )
                model_type = cfg.get("model_type", "").lower()
                # Check for abliteration marker
                if (
                    "abliteration" in str(cfg).lower()
                    or "heretic" in directory.name.lower()
                    or (directory.parent and "heretic" in directory.parent.name.lower())
                ):
                    kind = "abliterated"

                # Rough parameter estimation from standard transformer config
                hidden_size = cfg.get("hidden_size") or cfg.get("d_model")
                num_layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
                vocab_size = cfg.get("vocab_size", 32000)
                if hidden_size and num_layers:
                    # Standard causal LM parameter estimation formula
                    param_count = int(
                        num_layers * (12 * hidden_size**2) + (vocab_size * hidden_size)
                    )
        except Exception:
            pass

        fmt = "safetensors" if safetensor_files else ("pytorch" if bin_files else "unknown")
        vram_status, detail = estimate_vram_compatibility(param_count, size_mb, fmt)
        return LocalModelRecord(
            path=directory,
            name=directory.name,
            kind=kind,
            format=fmt,
            architecture=arch or model_type or "Transformer",
            parameter_count=param_count,
            size_mb=size_mb,
            vram_status=vram_status,
            status_detail=detail,
        )

    # 3. Check for standalone GGUF files in directory
    if gguf_files:
        vram_status, detail = estimate_vram_compatibility(None, size_mb, "gguf")
        return LocalModelRecord(
            path=directory,
            name=directory.name,
            kind="base",
            format="gguf",
            architecture="GGUF Model",
            parameter_count=None,
            size_mb=size_mb,
            vram_status=vram_status,
            status_detail=detail,
        )

    return None


class ModelService:
    """Manages model discovery, hub queries, and local library indexing."""

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

    def scan_local_models(self) -> list[LocalModelRecord]:
        """Scan models_dir and outputs_dir for local models and adapters."""
        records: list[LocalModelRecord] = []
        seen_paths: set[Path] = set()

        for label in ("models_dir", "outputs_dir"):
            validation = self._storage.validate(label, probe_writable=False)
            if not validation.exists or not validation.path.is_dir():
                continue
            root = validation.path
            # Check root itself if it's a model
            rec = inspect_model_directory(root)
            if rec and rec.path not in seen_paths:
                records.append(rec)
                seen_paths.add(rec.path)

            # Check immediate children and grandchild directories
            try:
                for entry in root.iterdir():
                    if entry.is_dir():
                        rec = inspect_model_directory(entry)
                        if rec and rec.path not in seen_paths:
                            records.append(rec)
                            seen_paths.add(rec.path)
                        # Also check one level deeper (e.g. outputs_dir/job_id/model)
                        for sub in entry.iterdir():
                            if sub.is_dir():
                                sub_rec = inspect_model_directory(sub)
                                if sub_rec and sub_rec.path not in seen_paths:
                                    records.append(sub_rec)
                                    seen_paths.add(sub_rec.path)
            except OSError as error:
                logger.warning("Error reading %s: %s", root, error)

        return sorted(records, key=lambda r: r.name.lower())

    def search_hub(self, query: str, limit: int = 20) -> list[HubModelRecord]:
        """Search Hugging Face Hub models."""
        from huggingface_hub import HfApi

        token = self._get_token()
        api = HfApi(token=token)
        try:
            results = list(
                api.list_models(
                    search=query.strip(),
                    limit=limit,
                    full=False,
                )
            )
            records: list[HubModelRecord] = []
            for item in results:
                model_id = getattr(item, "id", "") or getattr(item, "modelId", "")
                if not model_id:
                    continue
                parts = model_id.split("/")
                author = parts[0] if len(parts) > 1 else "community"
                name = parts[-1]
                records.append(
                    HubModelRecord(
                        repo_id=model_id,
                        author=author,
                        name=name,
                        downloads=int(getattr(item, "downloads", 0) or 0),
                        likes=int(getattr(item, "likes", 0) or 0),
                        pipeline_tag=str(getattr(item, "pipeline_tag", "") or "text-generation"),
                        private=bool(getattr(item, "private", False)),
                    )
                )
            return records
        except Exception as error:
            logger.error("Hub search failed: %s", error)
            raise

    def get_hub_model_info(self, repo_id: str) -> dict[str, Any]:
        """Get file listings and metadata for a hub repository."""
        from huggingface_hub import HfApi

        token = self._get_token()
        api = HfApi(token=token)
        try:
            info = api.model_info(repo_id=repo_id, files_metadata=True)
            files = []
            total_size = 0
            if hasattr(info, "siblings") and info.siblings:
                for file_info in info.siblings:
                    rfilename = getattr(file_info, "rfilename", "")
                    size = getattr(file_info, "size", 0) or 0
                    total_size += size
                    files.append({"filename": rfilename, "size": size})

            size_mb = round(total_size / (1024 * 1024), 1)
            vram_status, vram_detail = estimate_vram_compatibility(
                None,
                size_mb,
                "safetensors"
                if any(f["filename"].endswith(".safetensors") for f in files)
                else "other",
            )

            return {
                "repo_id": repo_id,
                "author": getattr(info, "author", "") or repo_id.split("/")[0],
                "pipeline_tag": getattr(info, "pipeline_tag", "") or "text-generation",
                "downloads": getattr(info, "downloads", 0) or 0,
                "likes": getattr(info, "likes", 0) or 0,
                "tags": getattr(info, "tags", []) or [],
                "files": files,
                "total_size_mb": size_mb,
                "vram_status": vram_status,
                "vram_detail": vram_detail,
            }
        except Exception as error:
            logger.error("Failed to get model info for %s: %s", repo_id, error)
            raise

    def download_model(
        self,
        repo_id: str,
        destination: Path | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        cancel: threading.Event | Callable[[], bool] | None = None,
    ) -> Path:
        """Download a model into ``destination`` one file at a time.

        Progress is reported as completed files over total files, not as an
        estimated byte percentage. ``cancel`` may be a ``threading.Event`` or
        a zero-argument callable; it is only checked *between* files. A file
        that has already been handed to ``hf_hub_download`` is allowed to
        finish, so this is cooperative cancellation and not true mid-file
        cancellation. Files downloaded before cancellation are retained.

        Raises ``DownloadCancelled`` when cancellation is observed, and
        re-raises any Hub error after logging it.
        """
        from huggingface_hub import HfApi, hf_hub_download

        token = self._get_token()
        if destination is None:
            models_dir = self._storage.path_for("models_dir")
            safe_folder = repo_id.replace("/", "--")
            destination = models_dir / safe_folder

        destination.mkdir(parents=True, exist_ok=True)

        def emit(fraction: float, message: str) -> None:
            if progress_callback:
                progress_callback(fraction, message)

        emit(0.0, f"Listing files for {repo_id}...")
        try:
            files = list(HfApi(token=token).list_repo_files(repo_id=repo_id, repo_type="model"))
        except Exception as error:
            logger.error("Failed to list files for %s: %s", repo_id, error)
            raise

        total = len(files)
        emit(0.0, f"Downloading {total} files for {repo_id}...")
        for index, filename in enumerate(files, start=1):
            completed = index - 1
            fraction = completed / total if total else 0.0
            if _is_cancelled(cancel):
                emit(
                    fraction,
                    f"Cancelling after {completed}/{total} files; "
                    "files already downloaded were kept.",
                )
                raise DownloadCancelled(destination, completed, total)

            emit(fraction, f"Downloading file {index}/{total}: {filename}")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                local_dir=str(destination),
                token=token,
            )
            emit(index / total if total else 0.0, f"Downloaded file {index}/{total}: {filename}")

        emit(1.0, f"Downloaded {repo_id} to {destination}")
        return destination
