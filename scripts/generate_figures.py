#!/usr/bin/env python3
"""Plot the saved LV fiber-P2 displacement trace against the published envelope."""

from pathlib import Path
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark import (
    COMPONENTS,
    POINTS,
    ROOT,
    component_rows,
    envelope,
    load_reference_curves,
    load_trace,
    metrics_dir,
)


def build_label(summary_path: Path) -> str:
    if not summary_path.is_file():
        return "provenance unavailable"
    summary = json.loads(summary_path.read_text())
    build = summary.get("solver_build")
    if not isinstance(build, dict):
        return "provenance unavailable"
    return f"tag {build['tag']}, commit {build['commit'][:8]}"


def main() -> None:
    trace_path = metrics_dir() / "displacement.json"
    if not trace_path.is_file():
        raise SystemExit(f"No displacement trace at {trace_path}. Extract a complete solve first.")
    trace = load_trace(trace_path)
    curves = load_reference_curves()
    # Raises if the trace does not cover the scored samples, so no figure is drawn from an incomplete run.
    component_rows(trace, curves)
    label = build_label(metrics_dir() / "summary.json")

    figure, axes = plt.subplots(3, 2, sharex=True, figsize=(9.6, 9.6))
    axes[0, 0].set_title(r"Particle $\mathbf{p}_0$")
    axes[0, 1].set_title(r"Particle $\mathbf{p}_1$")
    axes[0, 0].set_ylabel("Displacement x-component [m]")
    axes[1, 0].set_ylabel("Displacement y-component [m]")
    axes[2, 0].set_ylabel("Displacement z-component [m]")
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 1].set_xlabel("Time [s]")

    grid = np.arange(0.0, 1.0 + 1e-12, 0.01)
    for row_index, component in enumerate(COMPONENTS):
        for column, point in enumerate(POINTS):
            lower, upper = envelope(curves, grid, point, component)
            axis = axes[row_index, column]
            if row_index == 0 and column == 1:
                axis.fill_between(grid, lower, upper, color="0.75", zorder=0, label="benchmark range")
            else:
                axis.fill_between(grid, lower, upper, color="0.75", zorder=0)
            axis.plot(grid, lower, color="0.35", linewidth=1, zorder=1)
            axis.plot(grid, upper, color="0.35", linewidth=1, zorder=1)
            axis.plot(
                trace["time"],
                trace["displacement"][point][component],
                color="C0",
                linewidth=2.0,
                zorder=2,
                label="svMultiPhysics" if row_index == 0 and column == 0 else None,
            )
            axis.grid(True, color="0.85", linewidth=0.6)

    axes[0, 0].legend()
    axes[0, 1].legend()
    figure.suptitle(f"svMultiPhysics ({label})")
    figure.tight_layout()
    output = ROOT / "figures" / "displacement.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    print(output)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
