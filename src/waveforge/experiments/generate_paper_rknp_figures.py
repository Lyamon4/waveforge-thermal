"""Generate the complete paper-grade RKNP figure pack."""

from __future__ import annotations

import argparse
from pathlib import Path

from waveforge.reporting.paper_rknp import build_paper_figure_pack


def main() -> None:
    """Generate all registered figures and print the output location."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/paper_rknp_figure_pack"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir
    if not output_dir.is_absolute():
        output_dir = arguments.project_root / output_dir
    payload = build_paper_figure_pack(
        arguments.project_root,
        output_dir,
        dpi=arguments.dpi,
    )
    print(
        f"Generated {payload['figure_count']} paper figures in {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
