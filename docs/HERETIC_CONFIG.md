# Heretic 2.x configuration in the app

The Heretic tab drives the isolated Heretic 2.1+ engine with a real
`config.toml`; it is not a separate reimplementation of abliteration. The
configuration model lives in `src/services/heretic_config.py` and mirrors the
upstream `Settings` schema and plugin tables.

## Presets

`config/heretic_presets/` contains verbatim copies of the upstream example
configurations (`default`, `classifier`, `emotions`, `nohumor`, `noslop`,
`nosyco`, `piqa`, `terse`). The app adds a `Compact / low VRAM` profile that
uses 4-bit quantization, two trials and short prompt slices so the workflow
fits the reference 6 GB GTX 1660.

Loading a preset replaces the form. Nothing is run until **Run Heretic** is
pressed, and every value can be changed afterwards.

## Widgets and Raw TOML

The form covers every built-in Heretic setting that is useful in a desktop
workflow:

| Tab | Settings |
| --- | --- |
| Model & loading | dtypes, quantization, device map, max memory, offload, batch sizes, response length/prefix, system prompt, CoT blocks, debug output |
| Objectives | modifier plugin, scorer plugins and optimization direction, good/bad prompt sources, scorer prompt sources |
| Optimization | number of trials, startup trials, seed, checkpoint action, exported trial index |
| Export | merge vs adapter, local save vs upload, repository name/visibility, reproducibility information, shard size, tags |
| Raw TOML | complete `config.toml`, editable and validated in place |

The Raw TOML tab is authoritative for settings without widgets: keyword marker
lists, BERT classifier model/labels, benchmark task selection, custom plugin
import paths and any settings added by newer Heretic versions. Use
**Refresh from form** to re-render, **Apply TOML to form** to load edits, and
**Validate** to see problems before running. Unknown top-level keys and plugin
tables survive a round trip.

## Headless execution

At run time the app:

1. Reads the target model from the model combo.
2. Checks that at least one scorer optimizes a score and that the modifier list
   is valid.
3. Pins every value Heretic would otherwise prompt for (`export_strategy`,
   `checkpoint_action`, `trial_index`, `model_action`, output/checkpoint paths,
   and upload visibility/reproducibility for uploads).
4. Writes the resulting `config.toml` to a temporary file and starts
   `heretic_worker.py` in the Heretic environment with `HERETIC_CONFIG_FILE` set.
5. Streams trial progress and scores back over the JSON-lines protocol, and
   publishes the exported model to the Models tab when the job succeeds.

`HF_TOKEN` is passed to the worker only for upload jobs, using the token from
Windows Credential Manager. Heretic never receives a token for local saves.

## Not exposed

Heretic also supports interactive chat, benchmark runs after optimization, and
special modes (`--reproduce`, `--collect-reproducibles`, `--evaluate-model`).
These require an interactive terminal or are outside this app's scope; use the
Heretic CLI directly for them. Benchmark *scorers* (for example PIQA during
optimization) are fully supported through the scorer list.

## Attribution

Heretic is AGPL-3.0 © Philipp Emanuel Weidmann and contributors. The app invokes
it as an external tool in its own environment. Bundled preset files are data
copied from the upstream repository, with source details in
`config/heretic_presets/README.md`.
