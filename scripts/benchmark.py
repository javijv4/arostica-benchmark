#!/usr/bin/env python3
"""Extract, evaluate, and clean the LV fiber-P2 benchmark case.

The evaluation value is one number, in meters: how far the sampled
displacements sit outside the published envelope, clipped at zero when they
are inside. Diagnostics go to stderr.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "reference"
DT_S = 1e-3
SAMPLE_STEP = 10
N_STEPS = 1000
COMPONENTS = ("ux", "uy", "uz")
POINTS = ("p0", "p1")
SAMPLE_POINTS = {
    "p0": np.array([0.025, 0.03, 0.0]),
    "p1": np.array([0.0, 0.03, 0.0]),
}
REFERENCE_NAMES = (
    "monoventricular_nonblinded_step_1_group_carpentry.pickle",
    "monoventricular_nonblinded_step_1_group_ambit.pickle",
    "monoventricular_nonblinded_step_1_group_4c.pickle",
    "monoventricular_nonblinded_step_1_group_simula.pickle",
    "monoventricular_nonblinded_step_1_group_chimera.pickle",
    "monoventricular_nonblinded_step_1_group_cheart.pickle",
    "monoventricular_nonblinded_step_1_group_lifex.pickle",
    "monoventricular_nonblinded_step_1_group_simvascular_p1p1.pickle",
    "monoventricular_nonblinded_step_1_group_simvascular_p2.pickle",
    "monoventricular_nonblinded_step_1_group_comsol.pickle",
)
# Times at which a complete 0.01 s trace is compared. The published curves
# only overlap on 0.001 s to 0.999 s, so the 0 s and 1 s samples fall outside.
COMPARISON_TIMES = np.arange(1, 100) / 100.0


def case_dir() -> Path:
    return ROOT / "case"


def results_dir() -> Path:
    return case_dir() / "results"


def metrics_dir() -> Path:
    return case_dir() / "metrics"


def expected_steps() -> list[int]:
    return list(range(0, N_STEPS + 1, SAMPLE_STEP))


def snapshot_path(results: Path, step: int) -> Path:
    return results / f"result_{step:03d}.vtu"


def missing_steps(results: Path) -> list[int]:
    return [step for step in expected_steps() if not snapshot_path(results, step).is_file()]


def load_reference_curves() -> list[dict]:
    import pickle

    curves = []
    for name in REFERENCE_NAMES:
        path = REFERENCE_DIR / name
        with path.open("rb") as handle:
            curves.append(pickle.load(handle))
    return curves


def envelope(curves: list[dict], time: np.ndarray, point: str, component: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.vstack([
        np.interp(
            time,
            np.asarray(curve["time"], dtype=float),
            np.asarray(curve["displacement"][point][component], dtype=float),
        )
        for curve in curves
    ])
    return values.min(axis=0), values.max(axis=0)


def comparison_window(time: np.ndarray, curves: list[dict]) -> tuple[np.ndarray, float, float]:
    """Trace samples that lie inside every published curve."""
    span_start = max(float(np.asarray(curve["time"], dtype=float)[0]) for curve in curves)
    span_end = min(float(np.asarray(curve["time"], dtype=float)[-1]) for curve in curves)
    time = np.asarray(time, dtype=float)
    checked = time[(time >= span_start) & (time <= span_end)]
    return checked, span_start, span_end


def require_full_cycle(checked: np.ndarray) -> None:
    if checked.size == 0:
        raise SystemExit("Displacement trace has no samples inside the published time window.")
    missing = [
        float(sample)
        for sample in COMPARISON_TIMES
        if not np.any(np.isclose(checked, sample, rtol=0.0, atol=1e-12))
    ]
    if missing:
        preview = ", ".join(f"{sample:.2f}" for sample in missing[:5])
        raise SystemExit(
            "Displacement trace does not cover the published comparison samples "
            f"from 0.01 s to 0.99 s. Missing {len(missing)} samples, including {preview}."
        )


def component_rows(trace: dict, curves: list[dict]) -> tuple[list[dict], float, float]:
    time = np.asarray(trace["time"], dtype=float)
    checked, span_start, span_end = comparison_window(time, curves)
    require_full_cycle(checked)
    rows = []
    for point in POINTS:
        for component in COMPONENTS:
            lower, upper = envelope(curves, checked, point, component)
            values = np.interp(
                checked,
                time,
                np.asarray(trace["displacement"][point][component], dtype=float),
            )
            outside = np.maximum(lower - values, values - upper)
            worst = int(np.argmax(outside))
            rows.append({
                "point": point,
                "component": component,
                "max_outside_m": float(outside[worst]),
                "time_s": float(checked[worst]),
                "n_samples": int(checked.size),
            })
    return rows, span_start, span_end


def clipped_excursion(rows: list[dict]) -> float:
    """Meters outside the envelope. Zero when every sample is inside or on it."""
    return max(0.0, max(row["max_outside_m"] for row in rows))


def solver_build_from_env() -> dict | None:
    raw = os.environ.get("BENCHMARK_SOLVER_BUILD")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    keys = ("version", "tag", "commit")
    if not all(isinstance(data.get(key), str) and data[key] for key in keys):
        return None
    return {key: data[key] for key in keys}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def trace_from_samples(time: list[float], samples: list[np.ndarray]) -> dict:
    stacked = np.asarray(samples, dtype=float)
    return {
        "time_s": time,
        "sample_points_m": {name: SAMPLE_POINTS[name].tolist() for name in POINTS},
        "displacement_m": {
            name: {component: stacked[:, index, axis].tolist() for axis, component in enumerate(COMPONENTS)}
            for index, name in enumerate(POINTS)
        },
    }


def load_trace(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return {
        "time": np.asarray(payload["time_s"], dtype=float),
        "displacement": {
            point: {
                component: np.asarray(payload["displacement_m"][point][component], dtype=float)
                for component in COMPONENTS
            }
            for point in POINTS
        },
    }


def sample_snapshots(results: Path, steps: list[int]) -> dict:
    import pyvista as pv

    coordinates = np.vstack([SAMPLE_POINTS[name] for name in POINTS])
    times = [step / 1000.0 for step in steps]
    samples = []
    for step in steps:
        mesh = pv.read(snapshot_path(results, step))
        if "Displacement" not in mesh.point_data:
            raise SystemExit(f"{snapshot_path(results, step).name} has no Displacement point data.")
        sampled = pv.PolyData(coordinates).sample(mesh)
        displacement = np.asarray(sampled.point_data["Displacement"], dtype=float).reshape(len(POINTS), 3)
        if not np.all(np.isfinite(displacement)):
            raise SystemExit(f"{snapshot_path(results, step).name} produced a non-finite displacement sample.")
        samples.append(displacement)
    return trace_from_samples(times, samples)


def extract() -> None:
    results = results_dir()
    missing = missing_steps(results)
    if missing:
        preview = ", ".join(str(step) for step in missing[:8])
        raise SystemExit(
            f"Need result_*.vtu for steps 0, 10, ..., 1000 in {results}. "
            f"Missing {len(missing)} snapshots, including {preview}."
        )
    payload = sample_snapshots(results, expected_steps())
    build = solver_build_from_env()
    metrics = metrics_dir()
    write_json(metrics / "displacement.json", payload)
    write_json(metrics / "summary.json", {
        "solver_build": build,
        "provenance": "solver --version" if build is not None else "unavailable",
        "snapshots": len(expected_steps()),
        "time_start_s": 0.0,
        "time_end_s": 1.0,
        "sample_step": SAMPLE_STEP,
        "dt_s": DT_S,
    })


def write_diagnostics(rows: list[dict], value: float, span_start: float, span_end: float) -> None:
    print(f"comparison window [s]: {span_start:.3f} to {span_end:.3f}", file=sys.stderr)
    print(f"samples compared: {rows[0]['n_samples']}", file=sys.stderr)
    for row in rows:
        if row["max_outside_m"] <= 0.0:
            detail = f"inside by {-row['max_outside_m']:.6e} m"
        else:
            detail = f"outside by {row['max_outside_m']:.6e} m"
        print(
            f"  {row['point']} {row['component']}: {detail} at t = {row['time_s']:.3f} s",
            file=sys.stderr,
        )
    print(f"clipped excursion [m]: {value:.6e}", file=sys.stderr)


def evaluate(criterion: str) -> None:
    if criterion != "excursion":
        raise SystemExit(f"Unknown criterion {criterion!r}. Expected 'excursion'.")
    trace_path = metrics_dir() / "displacement.json"
    if not trace_path.is_file():
        raise SystemExit(f"No extracted displacement at {trace_path}. Run extract after a complete solve.")
    rows, span_start, span_end = component_rows(load_trace(trace_path), load_reference_curves())
    value = clipped_excursion(rows)
    if not np.isfinite(value):
        raise SystemExit("Clipped excursion is not finite.")
    write_diagnostics(rows, value, span_start, span_end)
    print(f"{value:.16g}")


def remove_raw_outputs(results: Path) -> None:
    """Delete VTK snapshots and restart files. Keep histor.dat and the directory."""
    if not results.is_dir():
        return
    for path in results.iterdir():
        if path.is_symlink() or not path.is_file():
            continue
        snapshot = path.name.startswith("result_") and path.suffix == ".vtu"
        restart = path.name.startswith("stFile_") and path.suffix == ".bin"
        if snapshot or restart:
            path.unlink()


def cleanup() -> None:
    remove_raw_outputs(results_dir())


def main(argv: list[str]) -> None:
    usage = (
        "usage: benchmark.py extract\n"
        "       benchmark.py evaluate excursion\n"
        "       benchmark.py cleanup"
    )
    if len(argv) < 2:
        raise SystemExit(usage)
    action = argv[1]
    rest = argv[2:]
    if action == "extract" and not rest:
        extract()
    elif action == "evaluate" and len(rest) == 1:
        evaluate(rest[0])
    elif action == "cleanup" and not rest:
        cleanup()
    else:
        raise SystemExit(usage)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except BrokenPipeError:
        raise SystemExit(0) from None
