"""Build the complete WaveForge MT3 frozen-test report and figure package."""

from __future__ import annotations

import argparse
from pathlib import Path

from waveforge.reporting.mt3_final import MT3FinalReportPaths, build_mt3_final_package


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    output = build_mt3_final_package(
        MT3FinalReportPaths(
            result_root=arguments.result_root.resolve(),
            training_root=arguments.training_root.resolve(),
            output_root=arguments.output.resolve(),
        ),
        dpi=arguments.dpi,
    )
    print(f"MT3_FINAL_PACKAGE={output}", flush=True)


if __name__ == "__main__":
    main()
