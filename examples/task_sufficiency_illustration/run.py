#!/usr/bin/env python3
"""Generate the task-sufficiency synthetic example."""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ocean-intelligence-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 6.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": "#A7B2BE",
    "axes.labelcolor": "#243447",
    "xtick.color": "#5B6878",
    "ytick.color": "#5B6878",
})

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT
FIGURES = ROOT

DEPTH_M = 500.0
N_DEPTH = 201
N_KERNELS = 12
N_RESOLVED_MODES = 4
RANGE_M = 20_000.0
TARGET_RMSE = 0.1
FREQUENCY_HZ = 500.0
ILL_CONDITIONING_EPS = 0.005
RESULT_NAME = "task_sufficiency_illustration.csv"
FIGURE_STEM = "task_sufficiency_illustration"


def build_operator(frequency_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Return depth and the differential-phase sensitivity operator K."""
    z = np.linspace(0.0, DEPTH_M, N_DEPTH)
    dz = z[1] - z[0]
    c = 1492.0 - 22.0 / (1.0 + np.exp(-(z - 120.0) / 18.0)) + 0.018 * z

    centers = np.linspace(50.0, 420.0, N_KERNELS)
    widths = np.linspace(45.0, 110.0, N_KERNELS)
    kernels = []
    for j, (center, width) in enumerate(zip(centers, widths)):
        base = np.exp(-0.5 * ((z - center) / width) ** 2)
        modulation = 1.0 + 0.25 * np.cos((j % 4 + 1) * np.pi * z / DEPTH_M + 0.3 * j)
        w = base * modulation
        w /= np.sum(w) * dz
        kernels.append(w)
    weights = np.asarray(kernels)

    sensitivity = -RANGE_M * weights * (dz / c**2)
    centering = np.eye(N_KERNELS) - np.ones((N_KERNELS, N_KERNELS)) / N_KERNELS
    operator = (2.0 * np.pi * frequency_hz / np.sqrt(N_KERNELS)) * centering @ sensitivity
    return z, operator


def scale_to_rmse(vector: np.ndarray, target: float = TARGET_RMSE) -> np.ndarray:
    rmse = np.linalg.norm(vector) / np.sqrt(vector.size)
    if rmse == 0:
        raise ValueError("cannot normalize a zero perturbation")
    return vector * (target / rmse)


def finite_transfer_factor(task: np.ndarray, metric: np.ndarray) -> tuple[float, np.ndarray]:
    """Return ||K M^{-1}||_2 and a state direction attaining the factor."""
    transfer = task @ np.linalg.inv(metric)
    _, singular_values, right_vectors_t = np.linalg.svd(transfer, full_matrices=False)
    measurement_direction = right_vectors_t[0]
    state_direction = np.linalg.solve(metric, measurement_direction)
    return float(singular_values[0]), state_direction


def run() -> tuple[list[dict[str, str]], dict[str, np.ndarray], np.ndarray]:
    z, task = build_operator(FREQUENCY_HZ)
    _, task_singular_values, task_right_vectors_t = np.linalg.svd(task, full_matrices=False)

    resolved_basis = task_right_vectors_t[:N_RESOLVED_MODES].T
    resolved_projector = resolved_basis @ resolved_basis.T
    unresolved_projector = np.eye(N_DEPTH) - resolved_projector

    full_metric = np.eye(N_DEPTH) / np.sqrt(N_DEPTH)
    ill_metric = (
        resolved_projector + ILL_CONDITIONING_EPS * unresolved_projector
    ) / np.sqrt(N_DEPTH)
    projected_metric = resolved_basis.T / np.sqrt(N_RESOLVED_MODES)

    lstar_full, full_worst = finite_transfer_factor(task, full_metric)
    lstar_ill, ill_worst = finite_transfer_factor(task, ill_metric)

    blind = task_right_vectors_t[N_RESOLVED_MODES]
    if task_singular_values[N_RESOLVED_MODES] <= 1e-12:
        raise RuntimeError("chosen projected counterexample is numerically task-null")

    directions = {
        "full_worst": scale_to_rmse(full_worst),
        "ill_worst": scale_to_rmse(ill_worst),
        "projected_blind": scale_to_rmse(blind),
    }

    phase = {name: float(np.linalg.norm(task @ vector)) for name, vector in directions.items()}
    metric_norm = {
        "full_worst": float(np.linalg.norm(full_metric @ directions["full_worst"])),
        "ill_worst": float(np.linalg.norm(ill_metric @ directions["ill_worst"])),
        "projected_blind": float(np.linalg.norm(projected_metric @ directions["projected_blind"])),
    }

    metrics: list[tuple[str, float | int, str]] = [
        ("illustrative_lstar_full", lstar_full, "rad_per_mps"),
        ("illustrative_lstar_near_blind", lstar_ill, "rad_per_mps"),
        ("illustrative_lstar_ratio_near_blind_to_full", lstar_ill / lstar_full, "ratio"),
        ("illustrative_task_response_full", phase["full_worst"], "rad"),
        ("illustrative_task_response_near_blind", phase["ill_worst"], "rad"),
        ("illustrative_task_response_projected_blind", phase["projected_blind"], "rad"),
        ("full_metric_worst_norm", metric_norm["full_worst"], "mps"),
        ("near_blind_metric_worst_norm", metric_norm["ill_worst"], "mps"),
        ("projected_metric_blind_norm", metric_norm["projected_blind"], "mps"),
        ("kernel_inclusion_full", 1, "boolean"),
        ("kernel_inclusion_ill_conditioned", 1, "boolean"),
        ("kernel_inclusion_projected", 0, "boolean"),
        ("ill_conditioning_epsilon", ILL_CONDITIONING_EPS, "ratio"),
        ("state_rmse", TARGET_RMSE, "mps"),
        ("n_depth", N_DEPTH, "count"),
        ("n_kernels", N_KERNELS, "count"),
        ("n_resolved_modes", N_RESOLVED_MODES, "count"),
        ("range_m", RANGE_M, "m"),
        ("frequency_hz", FREQUENCY_HZ, "Hz"),
    ]
    rows = [
        {"metric": name, "value": f"{value:.12g}", "unit": unit, "origin": "synthetic_analytical_illustration"}
        for name, value, unit in metrics
    ]
    return rows, directions, z


def write_results(rows: list[dict[str, str]]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / RESULT_NAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value", "unit", "origin"])
        writer.writeheader()
        writer.writerows(rows)


def write_figure(directions: dict[str, np.ndarray], z: np.ndarray, rows: list[dict[str, str]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    values = {row["metric"]: float(row["value"]) for row in rows}
    labels = ["Full-state", "Near-blind", "Projected blind"]
    keys = ["full_worst", "ill_worst", "projected_blind"]
    colors = ["#0072B2", "#D55E00", "#7A5195"]

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.28), gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]})
    for label, key, color in zip(labels, keys, colors):
        axes[0].plot(directions[key], z, label=label, color=color, linewidth=1.35)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(r"$\delta c$ (m/s)", fontsize=6.2)
    axes[0].set_ylabel("Depth (m)", fontsize=6.2)
    axes[0].set_title("A  Perturbations", loc="left", fontsize=7.1, fontweight="bold", color="#243447", pad=5)
    axes[0].grid(color="#D9E1E8", linewidth=0.45, alpha=0.8)
    axes[0].legend(fontsize=5.4, frameon=False, loc="lower left")

    phase_values = [values["illustrative_task_response_full"], values["illustrative_task_response_near_blind"], values["illustrative_task_response_projected_blind"]]
    axes[1].bar(range(3), phase_values, color=colors, width=0.68)
    axes[1].set_xticks(range(3), ["Full", "Near-\nblind", "Projected\nblind"], fontsize=5.7)
    axes[1].set_ylabel("Phase RMS (rad)", fontsize=6.2)
    axes[1].set_title("B  Task response", loc="left", fontsize=7.1, fontweight="bold", color="#243447", pad=5)
    axes[1].grid(axis="y", color="#D9E1E8", linewidth=0.45, alpha=0.8)

    finite = [values["illustrative_lstar_full"], values["illustrative_lstar_near_blind"]]
    axes[2].bar([0, 1], finite, color=colors[:2], width=0.65)
    axes[2].set_yscale("log")
    axes[2].set_xticks([0, 1, 2], ["Full", "Near-\nblind", "Projected"], fontsize=5.7)
    axes[2].set_xlim(-0.55, 2.55)
    axes[2].set_ylabel(r"$L^\star$ (rad/(m/s))", fontsize=6.2)
    axes[2].set_title("C  Two failure modes", loc="left", fontsize=7.1, fontweight="bold", color="#243447", pad=5)
    axes[2].grid(axis="y", color="#D9E1E8", linewidth=0.45, alpha=0.8, which="both")
    y_top = axes[2].get_ylim()[1]
    axes[2].annotate(
        r"$L^\star=\infty$",
        xy=(2, y_top * 0.78),
        ha="center",
        va="center",
        fontsize=6.5,
        color=colors[2],
        fontweight="bold",
    )
    axes[2].annotate(
        r"$\ker M\not\subseteq\ker K$",
        xy=(2, y_top * 0.43),
        ha="center",
        va="center",
        fontsize=5.2,
        color=colors[2],
    )

    for axis in axes:
        axis.tick_params(labelsize=5.6, length=2.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.22, wspace=0.47)
    for extension in ("pdf", "png"):
        path = FIGURES / f"{FIGURE_STEM}.{extension}"
        kwargs = {"dpi": 300, "bbox_inches": "tight", "pad_inches": 0.03}
        if extension == "pdf":
            kwargs["metadata"] = {"CreationDate": None, "ModDate": None}
        fig.savefig(path, **kwargs)
    plt.close(fig)


def main() -> None:
    rows, directions, z = run()
    write_results(rows)
    write_figure(directions, z, rows)
    print(f"Wrote {RESULTS / RESULT_NAME} and {FIGURE_STEM} figure")


if __name__ == "__main__":
    main()
