"""Run Heretic unattended when stdout is not a console.

Why this exists
---------------
Heretic builds ``questionary`` prompts *even for values that are already set* in
the configuration (see ``ask_if_unset(settings.trial_index, questionary.select(...))``
in ``heretic/main.py``). Constructing a prompt on Windows creates a
``prompt_toolkit`` ``Win32Output``, which calls ``GetConsoleScreenBufferInfo`` on
stdout and raises::

    NoConsoleScreenBufferError: Found xterm-256color, while expecting a
    Windows console.

That happens whenever stdout is redirected to a pipe - which is exactly what
``QProcess``, ``Tee-Object`` and CI do. The prompt is never *run* when every
interactive value is pinned in the config file, so giving prompt_toolkit a
``DummyOutput`` is enough to get past construction.

This launcher does not modify Heretic. It patches the output factory before
importing ``heretic.main`` and then calls Heretic's own entry point.

    $env:HERETIC_CONFIG_FILE = (Resolve-Path config\\heretic-verify.toml).Path
    .\\environments\\heretic\\Scripts\\python.exe scripts\\run_heretic_unattended.py

Arguments are forwarded unchanged, so ``--help`` and ``--model`` still work.
"""

from __future__ import annotations

import sys


def install_dummy_prompt_output() -> None:
    """Force prompt_toolkit to use a headless output."""
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.output import defaults as output_defaults

    def create_output(*args: object, **kwargs: object) -> DummyOutput:
        return DummyOutput()

    output_defaults.create_output = create_output  # type: ignore[assignment]


def main(argv: list[str] | None = None) -> int:
    install_dummy_prompt_output()

    from heretic.main import main as heretic_main

    sys.argv = ["heretic", *(sys.argv[1:] if argv is None else argv)]
    result = heretic_main()
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
