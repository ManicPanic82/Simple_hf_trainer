# Project roadmap

This roadmap preserves the six-tab product contract: **Models, Datasets, Training, Analytics, Heretic, Ollama Export**. It is an implementation and acceptance plan, not permission to expand the product into chat, agents, RAG, or hosted training.

## Phase 1 — Stabilise the desktop and worker foundation

- [x] Preserve the six-tab desktop shell and isolated engine environments.
- [x] Keep GPU ownership centralised so Unsloth and Heretic cannot run concurrently.
- [x] Keep worker protocol output separate from engine logs and provide bounded cancellation.
- [ ] Re-run lifecycle, queue, cancellation, restart and failure-path tests on Windows.
- [ ] Verify that a failed or killed worker always releases the GPU lock and leaves the UI usable.

**Exit criteria:** no orphaned worker, stuck GPU lock, or permanently disabled job controls across success, failure and cancellation.

## Phase 2 — Complete the Training workflow

- [x] Expose maximum steps or epoch-based training, warmup, optimizer, checkpoint strategy and evaluation strategy.
- [x] Expose optional evaluation-dataset selection and pass all selected settings to the real Unsloth worker.
- [ ] Verify the installed TRL/Transformers version accepts the selected configuration aliases.
- [ ] Run a small real QLoRA job with checkpoints and evaluation enabled.
- [ ] Verify adapter-only and merged-16-bit exports and confirm the original base model remains unchanged.

**Exit criteria:** every visible Training control affects the actual trainer configuration; invalid combinations are rejected before GPU allocation.

## Phase 3 — Verify Models

- [ ] Test Hub search, metadata, downloads, local scans and model registration.
- [ ] Verify incomplete downloads and corrupt model folders are reported and recoverable.
- [ ] Check architecture, quantization and approximate VRAM warnings are clearly labelled as estimates.
- [ ] Verify training and Heretic outputs appear in the local model library with correct types.

**Exit criteria:** users can discover, download, select and reuse models without overwriting source models.

## Phase 4 — Verify Datasets

- [ ] Test JSON, JSONL, CSV, Parquet and Arrow import/preview/prepare paths.
- [ ] Validate row schemas, empty datasets, malformed records and formatting errors before launching an engine.
- [ ] Verify train/evaluation dataset selection and split metadata.
- [ ] Keep source datasets immutable and make prepared outputs traceable.

**Exit criteria:** malformed or incompatible data is explained before a GPU job starts.

## Phase 5 — Verify Heretic

- [ ] Exercise the GUI-to-worker configuration hand-off, presets and Raw TOML round-trip.
- [ ] Verify trial metrics, console output, cancellation, output registration and export.
- [ ] Confirm failures leave useful logs and do not strand GPU ownership.
- [ ] Document the upstream Heretic AGPL-3.0-or-later relationship and distribution implications.

**Exit criteria:** a complete small abliteration can be configured, run, inspected and exported from the GUI.

## Phase 6 — Complete Analytics

- [x] Route real Unsloth metrics and loaded model architecture into the dedicated Analytics workspace.
- [x] Plot loss, evaluation loss, learning rate, gradient norm, throughput and hardware utilisation.
- [x] Summarise discovered datasets and show persisted terminal job history.
- [ ] Verify metric charts and hardware history during a real Windows training run.
- [ ] Add per-run metric persistence and comparison only after the live pipeline is accepted.

**Exit criteria:** Analytics displays only real measurements and clearly distinguishes live-session metrics from persisted job summaries.

## Phase 7 — Complete Ollama Export

- [x] Convert merged Hugging Face model folders to F16 GGUF using llama.cpp.
- [x] Offer Q4_K_M, Q5_K_M, Q8_0 and F16 export paths.
- [x] Import the resulting GGUF with Ollama and verify it with `ollama show`.
- [x] Protect source models and refuse to overwrite existing export artifacts.
- [ ] Run a full Windows conversion/import test with the selected llama.cpp and Ollama installations.
- [ ] Evaluate a safe adapter-merge workflow as a separate enhancement; adapter-only folders are currently rejected.

**Exit criteria:** a merged local Hugging Face model can be converted, imported and verified without modifying its source.

## Phase 8 — Windows acceptance and release preparation

- [ ] Run the complete pytest and Ruff suites in the application environment.
- [ ] Perform GUI acceptance on Windows 10/11 with the target 6 GB GPU.
- [ ] Test clean install, engine setup, restart behaviour and storage paths.
- [ ] Document verified model sizes, known limitations and reproducible acceptance steps.
- [ ] Prepare packaging only after the preceding acceptance checks pass.

**Exit criteria:** a repeatable Windows acceptance checklist passes. Do not merge the development PR or describe the app as release-ready before this point.

## Working rules

- Work on `fix/restore-four-tab-contract` while it remains the active development PR branch.
- Never make direct changes to `main`, merge the PR, or force-push as part of this roadmap.
- Prefer focused, tested fixes over broad refactors. Large UI modules can be decomposed after behaviour is stable.
- Tests that require real model downloads or GPU execution remain explicit manual acceptance tests; ordinary pytest must remain offline-friendly.
