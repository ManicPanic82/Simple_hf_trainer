"""Heretic 2.x configuration model, presets, and TOML (de)serialization.

The application environment deliberately does not install Heretic or Pydantic,
so this module models the subset of Heretic's ``Settings`` schema that the GUI
exposes, using only the standard library.  ``tomllib`` (Python 3.11+) reads any
``config.toml`` a user supplies, and :meth:`HereticConfig.to_toml` writes a
complete, self-contained configuration for the isolated Heretic worker.

Unknown top-level keys and plugin tables are preserved.  That keeps the GUI
usable with newer Heretic releases and with user-written scorer/modifier
plugins, while the widget form covers the built-in 2.x surface.

The field names and defaults mirror ``config.default.toml`` from
https://github.com/p-e-w/heretic (AGPL-3.0).  The upstream preset files are
bundled under ``config/heretic_presets/`` and loaded through :func:`load_preset`.
"""

from __future__ import annotations

import copy
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app import paths

MODIFIER_ARA = "heretic.modifiers.ara.ARA"
MODIFIER_ABLITERATION = "heretic.modifiers.abliteration.Abliteration"

SCORER_KEYWORD = "heretic.scorers.keyword_rate.KeywordRate"
SCORER_KL = "heretic.scorers.kl_divergence.KLDivergence"
SCORER_CLASSIFIER = "heretic.scorers.bert_classifier.BERTClassifier"
SCORER_BENCHMARK = "heretic.scorers.benchmark_score.BenchmarkScore"
SCORER_RESPONSE_LENGTH = "heretic.scorers.response_length.ResponseLength"

SCORER_LABELS = {
    "Refusals (keyword rate)": SCORER_KEYWORD,
    "KL divergence": SCORER_KL,
    "BERT classifier": SCORER_CLASSIFIER,
    "Benchmark score": SCORER_BENCHMARK,
    "Response length": SCORER_RESPONSE_LENGTH,
}
MODIFIER_LABELS = {
    "ARA (Heretic 2.x default)": MODIFIER_ARA,
    "Directional abliteration (classic)": MODIFIER_ABLITERATION,
}
OPTIMIZATIONS = ("minimize", "maximize", "none")
QUANTIZATIONS = ("none", "bnb_4bit")
EXPORT_STRATEGIES = ("merge", "adapter")
MODEL_ACTIONS = ("save", "upload")
CHECKPOINT_ACTIONS = ("restart", "continue")
REPRODUCIBILITY = ("full", "basic", "none")

_DATASET_KEYS = (
    "dataset",
    "commit",
    "config",
    "split",
    "column",
    "prefix",
    "suffix",
    "system_prompt",
)

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


class HereticConfigError(ValueError):
    """Raised when a configuration value or TOML document is invalid."""


@dataclass
class DatasetSpec:
    """One prompt source (``SingleDatasetSpecification`` in Heretic)."""

    dataset: str = ""
    split: str | None = None
    column: str | None = None
    prefix: str = ""
    suffix: str = ""
    system_prompt: str | None = None
    commit: str | None = None
    config: str | None = None

    @classmethod
    def from_value(cls, value: Any) -> DatasetSpec:
        if isinstance(value, str):
            return cls(dataset=value)
        if not isinstance(value, dict):
            raise HereticConfigError("dataset specification must be a table")
        unknown = set(value) - set(_DATASET_KEYS)
        if unknown:
            raise HereticConfigError(f"unknown dataset field(s): {', '.join(sorted(unknown))}")
        return cls(**{key: value[key] for key in _DATASET_KEYS if key in value})

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"dataset": self.dataset}
        for key in _DATASET_KEYS[1:]:
            value = getattr(self, key)
            if value is not None and value != "":
                result[key] = value
        return result

    def summary(self) -> str:
        parts = [self.dataset or "<unset>"]
        if self.split:
            parts.append(self.split)
        if self.column:
            parts.append(f"column={self.column}")
        return " / ".join(parts)


@dataclass
class PluginEntry:
    """A scorer or modifier entry from the ``scorers``/``modifiers`` list."""

    plugin: str
    optimization: str | None = None
    instance_name: str | None = None

    @classmethod
    def from_value(cls, value: Any, *, scorer: bool) -> PluginEntry:
        if not isinstance(value, dict) or not isinstance(value.get("plugin"), str):
            raise HereticConfigError("plugin entries must be tables with a 'plugin' string")
        allowed = {"plugin", "instance_name", "optimization" if scorer else "__unused__"}
        unknown = set(value) - allowed
        if unknown:
            raise HereticConfigError(f"unknown plugin field(s): {', '.join(sorted(unknown))}")
        return cls(
            plugin=value["plugin"],
            optimization=value.get("optimization") if scorer else None,
            instance_name=value.get("instance_name"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"plugin": self.plugin}
        if self.optimization is not None:
            result["optimization"] = self.optimization
        if self.instance_name:
            result["instance_name"] = self.instance_name
        return result


@dataclass
class HereticConfig:
    """A portable representation of a Heretic 2.x ``config.toml``."""

    model: str = ""
    model_commit: str | None = None
    dtypes: list[str] = field(default_factory=lambda: ["auto", "float16", "bfloat16", "float32"])
    quantization: str = "none"
    device_map: str = "auto"
    max_memory: dict[str, str] | None = None
    offload_outputs_to_cpu: bool = True
    batch_size: int = 0
    max_batch_size: int = 128
    batch_size_test_prompts: list[DatasetSpec] = field(
        default_factory=lambda: [DatasetSpec("mlabonne/harmless_alpaca", "train[:256]", "text")]
    )
    max_response_length: int = 100
    response_prefix: str | None = None
    response_prefix_test_prompts: list[DatasetSpec] = field(
        default_factory=lambda: [
            DatasetSpec("mlabonne/harmless_alpaca", "train[:100]", "text"),
            DatasetSpec("mlabonne/harmful_behaviors", "train[:100]", "text"),
        ]
    )
    chain_of_thought_blocks: list[list[str]] = field(
        default_factory=lambda: [
            ["<think>", "</think>"],
            [
                "<|channel|>analysis<|message|>",
                "<|end|><|start|>assistant<|channel|>final<|message|>",
            ],
            ["<thought>", "</thought>"],
            ["[THINK]", "[/THINK]"],
        ]
    )
    print_debug_information: bool = False
    scorers: list[PluginEntry] = field(
        default_factory=lambda: [
            PluginEntry(SCORER_KEYWORD, "minimize"),
            PluginEntry(SCORER_KL, "minimize"),
        ]
    )
    modifiers: list[PluginEntry] = field(default_factory=lambda: [PluginEntry(MODIFIER_ARA)])
    n_trials: int = 100
    n_startup_trials: int = 30
    seed: int | None = None
    study_checkpoint_dir: str = "checkpoints"
    modified_model_type: str = "decensored"
    benchmarks: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "task": "agieval",
                "name": "AGIEval",
                "description": "A Human-Centric Benchmark for Evaluating Foundation Models",
            },
            {
                "task": "bbh",
                "name": "BIG-Bench Hard (BBH)",
                "description": "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them",
            },
            {
                "task": "commonsense_qa",
                "name": "CommonsenseQA",
                "description": "A Question Answering Challenge Targeting Commonsense Knowledge",
            },
            {
                "task": "eq_bench",
                "name": "EQ-Bench",
                "description": "An Emotional Intelligence Benchmark for Large Language Models",
            },
            {
                "task": "gsm8k",
                "name": "GSM8K",
                "description": "Training Verifiers to Solve Math Word Problems",
            },
            {
                "task": "hellaswag",
                "name": "HellaSwag",
                "description": "Can a Machine Really Finish Your Sentence?",
            },
            {
                "task": "ifeval",
                "name": "IFEval",
                "description": "Instruction-Following Evaluation for Large Language Models",
            },
            {
                "task": "mmlu",
                "name": "MMLU",
                "description": "Measuring Massive Multitask Language Understanding",
            },
            {
                "task": "mmlu_pro",
                "name": "MMLU-Pro",
                "description": "A More Robust and Challenging Multi-Task Language Understanding Benchmark",
            },
            {
                "task": "piqa",
                "name": "PIQA",
                "description": "Reasoning about Physical Commonsense in Natural Language",
            },
            {
                "task": "winogrande",
                "name": "WinoGrande",
                "description": "An Adversarial Winograd Schema Challenge at Scale",
            },
        ]
    )
    max_shard_size: str | int = "5GB"
    export_strategy: str | None = None
    checkpoint_action: str | None = None
    trial_index: int | None = None
    n_additional_trials: int | None = None
    model_action: str | None = None
    save_directory: str | None = None
    upload_repo_id: str | None = None
    upload_repo_private: bool | None = None
    upload_reproducibility_information: str | None = None
    upload_tags: list[str] = field(
        default_factory=lambda: ["uncensored", "decensored", "abliterated"]
    )
    ignore_mismatches: bool | None = None
    system_prompt: str = "You are a helpful assistant."
    enable_shell: bool = False
    plugin_tables: dict[str, dict[str, Any]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    # -- construction -----------------------------------------------------

    @classmethod
    def default(cls) -> HereticConfig:
        """Upstream Heretic 2.x defaults (``config.default.toml``)."""
        cfg = cls()
        cfg.plugin_tables = copy.deepcopy(_UPSTREAM_PLUGIN_TABLES)
        return cfg

    @classmethod
    def compact(cls) -> HereticConfig:
        """A small, 6 GB-friendly starter configuration using classic abliteration."""
        cfg = cls.default()
        cfg.quantization = "bnb_4bit"
        cfg.batch_size = 2
        cfg.max_batch_size = 8
        cfg.max_response_length = 32
        cfg.n_trials = 2
        cfg.n_startup_trials = 1
        cfg.seed = 42
        cfg.modified_model_type = "abliterated"
        cfg.modifiers = [PluginEntry(MODIFIER_ABLITERATION)]
        cfg.response_prefix_test_prompts = [
            DatasetSpec("mlabonne/harmless_alpaca", "train[:5]", "text")
        ]
        cfg.plugin_tables = {
            "scorer": {
                "KeywordRate": {
                    "prompts": DatasetSpec(
                        "mlabonne/harmful_behaviors", "test[:5]", "text"
                    ).to_dict(),
                },
                "KLDivergence": {
                    "prompts": DatasetSpec(
                        "mlabonne/harmless_alpaca", "test[:5]", "text"
                    ).to_dict(),
                },
            },
            "modifier": {
                "Abliteration": {
                    "good_prompts": DatasetSpec(
                        "mlabonne/harmless_alpaca", "train[:5]", "text"
                    ).to_dict(),
                    "bad_prompts": DatasetSpec(
                        "mlabonne/harmful_behaviors", "train[:5]", "text"
                    ).to_dict(),
                },
            },
        }
        return cfg

    @classmethod
    def from_toml(cls, text: str, *, base: HereticConfig | None = None) -> HereticConfig:
        """Parse a Heretic ``config.toml``.

        Parsing starts from ``base`` when supplied.  With no base, only the
        dataclass defaults are used, which makes ``from_toml(cfg.to_toml())`` an
        exact round trip.  Callers that want Heretic's own "layer the file over
        the defaults" behaviour (presets, user-pasted partial configs) should
        pass ``base=HereticConfig.default()``.
        """
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as error:
            raise HereticConfigError(f"Invalid TOML: {error}") from error
        if not isinstance(data, dict):
            raise HereticConfigError("Configuration root must be a table")

        cfg = copy.deepcopy(base) if base is not None else cls()

        for name, cast in _SCALAR_CASTS.items():
            if name not in data:
                continue
            try:
                setattr(cfg, name, cast(data[name], name))
            except HereticConfigError:
                raise
            except (TypeError, ValueError) as error:
                raise HereticConfigError(f"{name}: {error}") from error

        if "batch_size_test_prompts" in data:
            cfg.batch_size_test_prompts = _as_datasets(
                data["batch_size_test_prompts"], "batch_size_test_prompts"
            )
        if "response_prefix_test_prompts" in data:
            cfg.response_prefix_test_prompts = _as_datasets(
                data["response_prefix_test_prompts"], "response_prefix_test_prompts"
            )
        if "scorers" in data:
            cfg.scorers = _as_plugins(data["scorers"], "scorers", scorer=True)
        if "modifiers" in data:
            cfg.modifiers = _as_plugins(data["modifiers"], "modifiers", scorer=False)

        for namespace in ("scorer", "modifier"):
            if namespace in data:
                value = data[namespace]
                if not isinstance(value, dict):
                    raise HereticConfigError(f"[{namespace}] must be a table")
                cfg.plugin_tables[namespace] = _deep_merge(
                    cfg.plugin_tables.get(namespace, {}), value
                )

        known = set(_SCALAR_CASTS) | {
            "batch_size_test_prompts",
            "response_prefix_test_prompts",
            "scorers",
            "modifiers",
            "scorer",
            "modifier",
        }
        for key, value in data.items():
            if key not in known:
                cfg.extra[key] = value
        return cfg

    # -- validation -------------------------------------------------------

    def validate(self) -> list[str]:
        """Return human-readable problems; an empty list means runnable."""
        errors: list[str] = []
        if not self.model.strip():
            errors.append("model is required")
        if self.n_trials < 1:
            errors.append("n_trials must be at least 1")
        if self.n_startup_trials < 0:
            errors.append("n_startup_trials cannot be negative")
        if self.batch_size < 0:
            errors.append("batch_size cannot be negative (0 means auto)")
        if self.max_batch_size < 1:
            errors.append("max_batch_size must be at least 1")
        if self.max_response_length < 1:
            errors.append("max_response_length must be at least 1")
        if self.quantization not in QUANTIZATIONS:
            errors.append(f"quantization must be one of {', '.join(QUANTIZATIONS)}")
        if len(self.modifiers) != 1:
            errors.append("Heretic currently requires exactly one modifier")
        if not any(entry.optimization in ("minimize", "maximize") for entry in self.scorers):
            errors.append("at least one scorer must be set to 'minimize' or 'maximize'")
        for entry in self.scorers:
            if entry.optimization not in OPTIMIZATIONS:
                errors.append(
                    f"scorer {entry.plugin}: optimization must be one of {', '.join(OPTIMIZATIONS)}"
                )
        if self.export_strategy is not None and self.export_strategy not in EXPORT_STRATEGIES:
            errors.append(f"export_strategy must be one of {', '.join(EXPORT_STRATEGIES)}")
        if self.checkpoint_action is not None and self.checkpoint_action not in CHECKPOINT_ACTIONS:
            errors.append(f"checkpoint_action must be one of {', '.join(CHECKPOINT_ACTIONS)}")
        if self.model_action is not None and self.model_action not in MODEL_ACTIONS:
            errors.append(
                "model_action must be 'save' or 'upload' (chat and benchmarks need "
                "an interactive Heretic session)"
            )
        if self.model_action == "upload" and not (self.upload_repo_id or "").strip():
            errors.append("upload_repo_id is required when model_action is 'upload'")
        if (
            self.upload_reproducibility_information is not None
            and self.upload_reproducibility_information not in REPRODUCIBILITY
        ):
            errors.append(
                f"upload_reproducibility_information must be one of {', '.join(REPRODUCIBILITY)}"
            )
        if self.trial_index is not None and self.trial_index < 0:
            errors.append("trial_index cannot be negative")
        if not self.response_prefix_test_prompts:
            errors.append("at least one response_prefix_test_prompts source is required")
        for name, specs in (
            ("batch_size_test_prompts", self.batch_size_test_prompts),
            ("response_prefix_test_prompts", self.response_prefix_test_prompts),
        ):
            if any(not spec.dataset.strip() for spec in specs):
                errors.append(f"{name}: dataset is required")
        return errors

    def headless_copy(self, output_dir: Path, checkpoints_dir: Path) -> HereticConfig:
        """Pin every value Heretic would otherwise ask for interactively."""
        result = copy.deepcopy(self)
        result.model_action = self.model_action or "save"
        result.export_strategy = self.export_strategy or "merge"
        result.checkpoint_action = self.checkpoint_action or "restart"
        result.trial_index = self.trial_index if self.trial_index is not None else 0
        result.save_directory = str(output_dir)
        result.study_checkpoint_dir = str(checkpoints_dir)
        # Upload jobs must not stop at an interactive question either.
        result.upload_repo_private = (
            self.upload_repo_private if self.upload_repo_private is not None else False
        )
        result.upload_reproducibility_information = (
            self.upload_reproducibility_information or "none"
        )
        return result

    # -- serialization ----------------------------------------------------

    def to_toml(self) -> str:
        """Serialize the full configuration as a valid ``config.toml``."""
        lines: list[str] = [
            "# Generated by Simple HF Trainer for Heretic 2.x.",
            "# Edit freely and use 'Apply TOML to form' to load your changes.",
            "",
        ]
        for name, value in _toml_fields(self):
            if value is None:
                continue
            rendered = _toml_value(value)
            if rendered is not None:
                lines.append(f"{name} = {rendered}")

        for name, specs in (
            ("batch_size_test_prompts", self.batch_size_test_prompts),
            ("response_prefix_test_prompts", self.response_prefix_test_prompts),
        ):
            if specs:
                lines.append("")
                for spec in specs:
                    lines.append(f"[[{name}]]")
                    for key, value in spec.to_dict().items():
                        lines.append(f"{key} = {_toml_value(value)}")

        for key, value in sorted(self.extra.items()):
            _write_table(lines, key, value)

        for namespace in ("scorer", "modifier"):
            tables = self.plugin_tables.get(namespace)
            if isinstance(tables, dict) and tables:
                _write_table(lines, namespace, tables)

        return "\n".join(lines).rstrip() + "\n"

    def modifier_class_name(self) -> str:
        """Class name used for ``[modifier.<ClassName>]`` plugin tables."""
        return self.modifiers[0].plugin.rsplit(".", 1)[-1] if self.modifiers else "ARA"


def _toml_fields(cfg: HereticConfig) -> list[tuple[str, Any]]:
    return [
        ("model", cfg.model),
        ("model_commit", cfg.model_commit),
        ("dtypes", cfg.dtypes),
        ("quantization", cfg.quantization),
        ("device_map", cfg.device_map),
        ("max_memory", cfg.max_memory),
        ("offload_outputs_to_cpu", cfg.offload_outputs_to_cpu),
        ("batch_size", cfg.batch_size),
        ("max_batch_size", cfg.max_batch_size),
        ("max_response_length", cfg.max_response_length),
        ("response_prefix", cfg.response_prefix),
        ("chain_of_thought_blocks", cfg.chain_of_thought_blocks),
        ("print_debug_information", cfg.print_debug_information),
        ("scorers", [entry.to_dict() for entry in cfg.scorers]),
        ("modifiers", [entry.to_dict() for entry in cfg.modifiers]),
        ("n_trials", cfg.n_trials),
        ("n_startup_trials", cfg.n_startup_trials),
        ("seed", cfg.seed),
        ("study_checkpoint_dir", cfg.study_checkpoint_dir),
        ("modified_model_type", cfg.modified_model_type),
        ("benchmarks", cfg.benchmarks),
        ("max_shard_size", cfg.max_shard_size),
        ("export_strategy", cfg.export_strategy),
        ("checkpoint_action", cfg.checkpoint_action),
        ("trial_index", cfg.trial_index),
        ("n_additional_trials", cfg.n_additional_trials),
        ("model_action", cfg.model_action),
        ("save_directory", cfg.save_directory),
        ("upload_repo_id", cfg.upload_repo_id),
        ("upload_repo_private", cfg.upload_repo_private),
        ("upload_reproducibility_information", cfg.upload_reproducibility_information),
        ("upload_tags", cfg.upload_tags),
        ("ignore_mismatches", cfg.ignore_mismatches),
        ("system_prompt", cfg.system_prompt),
        ("enable_shell", cfg.enable_shell),
    ]


def _toml_key(key: str) -> str:
    return key if _BARE_KEY.fullmatch(key) else json.dumps(str(key), ensure_ascii=False)


def _toml_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        items = [item for item in (_toml_value(entry) for entry in value) if item is not None]
        return "[" + ", ".join(items) + "]"
    if isinstance(value, dict):
        items = [
            f"{_toml_key(str(key))} = {rendered}"
            for key, raw in value.items()
            if (rendered := _toml_value(raw)) is not None
        ]
        return "{ " + ", ".join(items) + " }"
    return json.dumps(str(value), ensure_ascii=False)


def _write_table(lines: list[str], path: str, table: dict[str, Any]) -> None:
    simple: list[tuple[str, Any]] = []
    children: list[tuple[str, Any]] = []
    for key, value in table.items():
        if isinstance(value, dict) or (
            isinstance(value, list) and value and all(isinstance(v, dict) for v in value)
        ):
            children.append((key, value))
        else:
            simple.append((key, value))
    if simple or not children:
        lines.append("")
        lines.append(f"[{path}]")
        for key, value in simple:
            rendered = _toml_value(value)
            if rendered is not None:
                lines.append(f"{_toml_key(str(key))} = {rendered}")
    for key, value in children:
        child_path = f"{path}.{_toml_key(str(key))}"
        if isinstance(value, dict):
            _write_table(lines, child_path, value)
        else:
            for entry in value:
                lines.append("")
                lines.append(f"[[{child_path}]]")
                for sub_key, sub_value in entry.items():
                    rendered = _toml_value(sub_value)
                    if rendered is not None:
                        lines.append(f"{_toml_key(str(sub_key))} = {rendered}")


def _as_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise HereticConfigError(f"{name} must be a string")
    return value


def _as_optional_str(value: Any, name: str) -> str | None:
    return None if value is None else _as_str(value, name)


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HereticConfigError(f"{name} must be an integer")
    return value


def _as_optional_int(value: Any, name: str) -> int | None:
    return None if value is None else _as_int(value, name)


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise HereticConfigError(f"{name} must be a boolean")
    return value


def _as_optional_bool(value: Any, name: str) -> bool | None:
    return None if value is None else _as_bool(value, name)


def _as_str_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise HereticConfigError(f"{name} must be an array of strings")
    return list(value)


def _as_datasets(value: Any, name: str) -> list[DatasetSpec]:
    values = value if isinstance(value, list) else [value]
    return [DatasetSpec.from_value(item) for item in values]


def _as_plugins(value: Any, name: str, *, scorer: bool) -> list[PluginEntry]:
    if not isinstance(value, list):
        raise HereticConfigError(f"{name} must be an array of plugin tables")
    return [PluginEntry.from_value(item, scorer=scorer) for item in value]


def _as_cot_blocks(value: Any, name: str) -> list[list[str]]:
    if not isinstance(value, list):
        raise HereticConfigError(f"{name} must be an array of [initializer, finalizer] pairs")
    blocks: list[list[str]] = []
    for block in value:
        if (
            not isinstance(block, list)
            or len(block) != 2
            or not all(isinstance(item, str) for item in block)
        ):
            raise HereticConfigError(f"{name} entries must be [initializer, finalizer] strings")
        blocks.append(list(block))
    return blocks


def _as_max_memory(value: Any, name: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise HereticConfigError(f"{name} must be a table of device -> memory strings")
    return dict(value)


def _as_shard_size(value: Any, name: str) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise HereticConfigError(f"{name} must be a string or integer")
    return value


def _as_benchmarks(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise HereticConfigError(f"{name} must be an array of benchmark tables")
    return [dict(item) for item in value]


_SCALAR_CASTS = {
    "model": _as_str,
    "model_commit": _as_optional_str,
    "dtypes": _as_str_list,
    "quantization": _as_str,
    "device_map": _as_str,
    "max_memory": _as_max_memory,
    "offload_outputs_to_cpu": _as_bool,
    "batch_size": _as_int,
    "max_batch_size": _as_int,
    "max_response_length": _as_int,
    "response_prefix": _as_optional_str,
    "chain_of_thought_blocks": _as_cot_blocks,
    "print_debug_information": _as_bool,
    "n_trials": _as_int,
    "n_startup_trials": _as_int,
    "seed": _as_optional_int,
    "study_checkpoint_dir": _as_str,
    "modified_model_type": _as_str,
    "benchmarks": _as_benchmarks,
    "max_shard_size": _as_shard_size,
    "export_strategy": _as_optional_str,
    "checkpoint_action": _as_optional_str,
    "trial_index": _as_optional_int,
    "n_additional_trials": _as_optional_int,
    "model_action": _as_optional_str,
    "save_directory": _as_optional_str,
    "upload_repo_id": _as_optional_str,
    "upload_repo_private": _as_optional_bool,
    "upload_reproducibility_information": _as_optional_str,
    "upload_tags": _as_str_list,
    "ignore_mismatches": _as_optional_bool,
    "system_prompt": _as_str,
    "enable_shell": _as_bool,
}


_UPSTREAM_PLUGIN_TABLES: dict[str, dict[str, Any]] = {
    "scorer": {
        "KeywordRate": {
            "score_name": "Refusals",
            "print_responses": False,
            "keyword_markers": [
                "disclaimer",
                "sorry",
                "i can'",
                "i cant",
                "i cannot",
                "i won'",
                "i wont",
                "i will not",
                "i unable",
                "im unable",
                "i'm unable",
                "i am unable",
                "i an ai",
                "im an ai",
                "i'm an ai",
                "i am an ai",
                "as an ai",
                "ai assistant",
                "i designed to",
                "im designed to",
                "i'm designed to",
                "i am designed to",
                "i programmed to",
                "im programmed to",
                "i'm programmed to",
                "i am programmed to",
                "violat",
                "prohibit",
                "illegal",
                "harmful",
                "inappropriate",
                "unethical",
                "ethical boundaries",
            ],
            "prompts": {
                "dataset": "mlabonne/harmful_behaviors",
                "split": "test[:100]",
                "column": "text",
            },
        },
        "KLDivergence": {
            "prompts": {
                "dataset": "mlabonne/harmless_alpaca",
                "split": "test[:100]",
                "column": "text",
            }
        },
    },
    "modifier": {
        "ARA": {
            "preserve_row_magnitudes": True,
            "lora_rank": 50,
            "n_optimization_steps": 5,
            "learning_rate": 1.0,
            "max_iter": 20,
            "history_size": 10,
            "print_loss": False,
            "good_prompts": {
                "dataset": "mlabonne/harmless_alpaca",
                "split": "train[:400]",
                "column": "text",
            },
            "bad_prompts": {
                "dataset": "mlabonne/harmful_behaviors",
                "split": "train[:400]",
                "column": "text",
            },
        },
        "Abliteration": {
            "orthogonalize_direction": True,
            "row_normalization": "full",
            "full_normalization_lora_rank": 3,
            "winsorization_quantile": 1.0,
            "good_prompts": {
                "dataset": "mlabonne/harmless_alpaca",
                "split": "train[:400]",
                "column": "text",
            },
            "bad_prompts": {
                "dataset": "mlabonne/harmful_behaviors",
                "split": "train[:400]",
                "column": "text",
            },
        },
    },
}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# -- presets --------------------------------------------------------------

PRESET_LABELS = {
    "compact": "Compact / low VRAM (starter)",
    "default": "Upstream defaults (decensorship, ARA)",
    "classifier": "Refusal classifier (BERT)",
    "emotions": "Emotion steering (classifier)",
    "nohumor": "Reduce humor",
    "noslop": "Reduce slop",
    "nosyco": "Reduce sycophancy",
    "piqa": "Refusals + PIQA benchmark",
    "terse": "Terse responses",
}


def presets_dir() -> Path:
    return paths.repository_root() / "config" / "heretic_presets"


def available_presets() -> list[tuple[str, str]]:
    """Return ``(id, label)`` pairs; the built-in compact profile comes first."""
    result = [("compact", PRESET_LABELS["compact"])]
    directory = presets_dir()
    if directory.is_dir():
        for path in sorted(directory.glob("*.toml")):
            preset_id = path.stem
            result.append(
                (preset_id, PRESET_LABELS.get(preset_id, preset_id.replace("_", " ").title()))
            )
    return result


def load_preset(preset_id: str) -> HereticConfig:
    if preset_id == "compact":
        return HereticConfig.compact()
    path = presets_dir() / f"{preset_id}.toml"
    if not path.is_file():
        raise HereticConfigError(f"Unknown preset: {preset_id}")
    return HereticConfig.from_toml(path.read_text(encoding="utf-8"), base=HereticConfig.default())
