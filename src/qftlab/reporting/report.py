"""Portable plots and a self-contained, run-specific HTML summary."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qftlab.circuits.qft import PARAMETER_NAMES, QFT_PARAMETERS
from qftlab.evaluation.metrics import wrap_angle


def write_report(run_dir: Path, history: list[dict], frequency_targets: np.ndarray,
                 frequency_probabilities: np.ndarray, parameters: dict, summary: dict) -> list[Path]:
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    steps = [row["step"] for row in history]
    losses = [row["loss"] for row in history]
    fidelities = [row["process_fidelity"] for row in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, losses, color="#3157a4", linewidth=2)
    ax.set(xlabel="Optimizer iteration", ylabel="1 − process fidelity", title="QFT recovery loss")
    ax.grid(alpha=.25)
    fig.tight_layout()
    loss_path = artifacts / "loss.png"
    fig.savefig(loss_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, fidelities, color="#17805b", linewidth=2)
    ax.axhline(1.0, color="#777", linestyle="--", linewidth=1)
    ax.set(xlabel="Optimizer iteration", ylabel="Process fidelity", ylim=(0, 1.02), title="Whole-unitary recovery")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fidelity_path = artifacts / "process_fidelity.png"
    fig.savefig(fidelity_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
    bins = np.arange(frequency_probabilities.shape[1])
    for row, ax in enumerate(axes.flat):
        ax.bar(bins - .18, frequency_targets[row], .36, label="Ideal", color="#78a6d8")
        ax.bar(bins + .18, frequency_probabilities[row], .36, label="Learned", color="#e49b52")
        ax.set(title=f"Input frequency {row}", xlabel="Output bin", ylim=(0, 1.05), xticks=bins)
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].set_ylabel("Probability")
    axes[1, 0].set_ylabel("Probability")
    axes[0, 0].legend()
    fig.suptitle("Held-out frequency output distributions")
    fig.tight_layout()
    frequency_path = artifacts / "frequency_distributions.png"
    fig.savefig(frequency_path, dpi=160)
    plt.close(fig)

    confusion = np.zeros((frequency_probabilities.shape[1], frequency_probabilities.shape[1]), dtype=int)
    for expected, predicted in enumerate(np.argmax(frequency_probabilities, axis=1)):
        confusion[expected, predicted] += 1
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(confusion, cmap="Blues")
    ax.set(xlabel="Predicted frequency bin", ylabel="Expected frequency bin", title="Frequency confusion matrix",
           xticks=np.arange(confusion.shape[1]), yticks=np.arange(confusion.shape[0]))
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            ax.text(col, row, str(confusion[row, col]), ha="center", va="center",
                    color="white" if confusion[row, col] > confusion.max() / 2 else "#202733")
    fig.tight_layout()
    confusion_path = artifacts / "frequency_confusion.png"
    fig.savefig(confusion_path, dpi=160)
    plt.close(fig)

    parameter_keys = [f"parameter_{name}" for name in PARAMETER_NAMES]
    if history and all(key in history[0] for key in parameter_keys):
        trajectory = np.asarray([[row[key] for key in parameter_keys] for row in history])
        angle_error = wrap_angle(trajectory - QFT_PARAMETERS[None, :])
        fig, ax = plt.subplots(figsize=(9, 5))
        for index, name in enumerate(PARAMETER_NAMES):
            ax.plot(steps, angle_error[:, index], label=name)
        ax.set(xlabel="Optimizer iteration", ylabel="Wrapped angle error (radians)",
               title="Trainable parameter trajectory relative to canonical QFT")
        ax.grid(alpha=.25)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        parameters_path = artifacts / "parameter_trajectory.png"
        fig.savefig(parameters_path, dpi=160)
        plt.close(fig)
    else:
        parameters_path = None

    report = artifacts / "report.html"
    score_rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items() if isinstance(value, (str, int, float, bool))
    )
    report.write_text(
        "<!doctype html><meta charset='utf-8'><title>QFT recovery run</title>"
        "<style>body{font:16px system-ui;max-width:1050px;margin:2rem auto;padding:0 1rem;color:#202733}"
        "img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{padding:.45rem .8rem;border-bottom:1px solid #ddd;text-align:left}"
        ".plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1rem}</style>"
        f"<h1>QFT recovery</h1><p>Run: <code>{html.escape(run_dir.name)}</code></p>"
        f"<table>{score_rows}</table><div class='plots'><img src='loss.png'><img src='process_fidelity.png'>"
        "<img src='frequency_distributions.png'><img src='frequency_confusion.png'>"
        "<img src='parameter_trajectory.png'></div><h2>Final parameters</h2><pre>"
        f"{html.escape(json.dumps(parameters, indent=2))}</pre>",
        encoding="utf-8",
    )
    paths = [loss_path, fidelity_path, frequency_path, confusion_path, report]
    if parameters_path:
        paths.append(parameters_path)
    return paths
