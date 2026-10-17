"""Reproducible aggregate plots for learned circuit parameters.

All plots are regenerated from saved run summaries; no training is performed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qftlab.circuits.qft import PARAMETER_NAMES


COMPONENT_COUNTS = (2, 3, 4)
GROUP_COLORS = {2: "#0072B2", 3: "#D55E00", 4: "#009E73"}


def wrapped(angle: float) -> float:
    """Return the shortest signed representative in [-pi, pi)."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def categorical_run_colors(runs: list[dict]) -> dict[str, str]:
    """Assign a distinct categorical color within each component-count group.

    The ten-color palette is deliberately reused for each group: seed 42, for
    example, receives the same color for 2, 3, and 4 frequency targets.
    """
    palette = plt.get_cmap("tab10")
    colors: dict[str, str] = {}
    for n_components in COMPONENT_COUNTS:
        subset = sorted(
            (run for run in runs if run["n_components"] == n_components),
            key=lambda run: (run.get("seed", 0), run["run_id"]),
        )
        for index, run in enumerate(subset):
            colors[run["run_id"]] = matplotlib.colors.to_hex(palette(index % 10))
    return colors


def _write_histograms(runs: list[dict], output_dir: Path) -> None:
    # Grouped bars prevent identical/near-identical bins from hiding one another.
    bins = np.linspace(-np.pi, np.pi, 11)
    width = np.diff(bins)[0] / 3.25
    offsets = {2: -width, 3: 0.0, 4: width}
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)
    for index, name in enumerate(PARAMETER_NAMES):
        ax = axes.flat[index]
        for n_components in COMPONENT_COUNTS:
            values = [r["wrapped_angle_change_radians"][name] for r in runs
                      if r["n_components"] == n_components]
            counts, edges = np.histogram(values, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2
            ax.bar(centers + offsets[n_components], counts, width=width * .94,
                   color=GROUP_COLORS[n_components], edgecolor="white", linewidth=.45,
                   alpha=.9, label=f"{n_components} frequencies", zorder=3)
        ax.set_title(name)
        ax.grid(axis="y", alpha=.25, zorder=0)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                      ["−π", "−π/2", "0", "π/2", "π"])
        if index == 0:
            ax.legend(fontsize=8)
        if index // 4 == 1:
            ax.set_xlabel("Shortest signed Δθ (rad)")
        if index % 4 == 0:
            ax.set_ylabel("Run count")
    axes.flat[-1].axis("off")
    n_each = {n: sum(r["n_components"] == n for r in runs) for n in COMPONENT_COUNTS}
    fig.suptitle("Shortest signed angular changes by gate · grouped by target complexity")
    fig.text(.5, .015, "Side-by-side bars preserve all three counts when bins overlap. "
             + " · ".join(f"{n} frequencies: {count} runs" for n, count in n_each.items()),
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .97))
    fig.savefig(output_dir / "parameter_change_histograms.png", dpi=190)
    plt.close(fig)


def _write_spider_profiles(runs: list[dict], color_by_run: dict[str, str], output_dir: Path) -> None:
    spokes = np.linspace(0, 2 * np.pi, len(PARAMETER_NAMES), endpoint=False)
    closed_spokes = np.r_[spokes, spokes[0]]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": "polar"})
    for ax, n_components in zip(axes, COMPONENT_COUNTS):
        subset = [run for run in runs if run["n_components"] == n_components]
        ax.set_xticks(spokes, PARAMETER_NAMES, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .5, 1], labels=["−π", "0", "+π"], fontsize=8)
        ax.set_title(f"{n_components} frequencies\ninitial dashed · final solid", pad=18)
        for run in subset:
            color = color_by_run[run["run_id"]]
            initial = np.array([wrapped(run["initial_parameters"][name]) for name in PARAMETER_NAMES])
            final = np.array([wrapped(run["final_parameters"][name]) for name in PARAMETER_NAMES])
            initial_r = np.r_[(initial + np.pi) / (2 * np.pi), (initial[0] + np.pi) / (2 * np.pi)]
            final_r = np.r_[(final + np.pi) / (2 * np.pi), (final[0] + np.pi) / (2 * np.pi)]
            ax.plot(closed_spokes, initial_r, color=color, linestyle="--", marker="o",
                    markersize=2.0, linewidth=1.95, alpha=.43, zorder=2)
            ax.plot(closed_spokes, final_r, color=color, linestyle="-", marker="s",
                    markersize=2.1, linewidth=2.4, alpha=.63, zorder=3)
            for angle, r0, r1 in zip(spokes, initial_r[:-1], final_r[:-1]):
                ax.plot([angle, angle], [r0, r1], color=color, linewidth=1.95, alpha=.55, zorder=4)
    fig.suptitle("Seven-angle circuit profiles · run colors repeat by initialization across target groups", y=1.02)
    fig.text(.5, -.005, "Dense static overview; use the per-gate circles and run-level parameter plots for detail.",
             ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "parameter_spider_profiles.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def _write_circular_changes(runs: list[dict], color_by_run: dict[str, str], output_dir: Path) -> list[str]:
    outputs = []
    for name in PARAMETER_NAMES:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), subplot_kw={"projection": "polar"})
        for ax, n_components in zip(axes, COMPONENT_COUNTS):
            subset = [run for run in runs if run["n_components"] == n_components]
            ax.set_ylim(0, 1.18)
            ax.set_rticks([])
            ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 4),
                          ["0", "π/4", "π/2", "3π/4", "π", "−3π/4", "−π/2", "−π/4"], fontsize=8)
            ax.set_title(f"{n_components} frequencies")
            endpoints = []
            for run in subset:
                color = color_by_run[run["run_id"]]
                theta0 = wrapped(run["initial_parameters"][name])
                theta1 = wrapped(run["final_parameters"][name])
                delta = wrapped(theta1 - theta0)
                # Draw the shortest circular arc above the polar grid; use an
                # actual arc (rather than a chord) so angle direction is legible.
                if abs(delta) > 1e-10:
                    arc = np.linspace(theta0, theta0 + delta, max(8, int(abs(delta) * 24)))
                    ax.plot(arc, np.ones_like(arc), color=color, linewidth=2.0,
                            alpha=.88, solid_capstyle="round", zorder=3)
                endpoints.append((color, theta0, theta1))
            # Draw every connector before any endpoint. This guarantees that no
            # other run's line can cross over and hide a square or circle.
            # Marker edges separate endpoints from both the connectors and grid.
            for color, theta0, _ in endpoints:
                ax.scatter([theta0], [1], facecolors="white", edgecolors=[color],
                           linewidths=2.0, marker="o", s=76, zorder=10)
            for color, _, theta1 in endpoints:
                ax.scatter([theta1], [1], facecolors=[color], edgecolors=["#202020"],
                           linewidths=1.25, marker="s", s=52, zorder=11)
        fig.suptitle(f"{name}: shortest signed initial-to-final angular change", y=1.01)
        fig.tight_layout()
        filename = f"{name}_circular_changes.png"
        fig.savefig(output_dir / filename, dpi=190, bbox_inches="tight")
        plt.close(fig)
        outputs.append(filename)
    return outputs


def generate_parameter_summary_plots(runs: list[dict], color_by_run: dict[str, str], output_dir: Path) -> list[str]:
    """Create all reusable aggregate parameter plots and describe their methods."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_histograms(runs, output_dir)
    _write_spider_profiles(runs, color_by_run, output_dir)
    outputs = ["parameter_change_histograms.png", "parameter_spider_profiles.png"]
    outputs.extend(_write_circular_changes(runs, color_by_run, output_dir))
    metadata = {
        "generator": "qftlab.visualization.parameter_summaries.generate_parameter_summary_plots",
        "plot_version": 3,
        "input": "saved summary.json run records (initial/final parameter values and wrapped angle changes)",
        "methods": {
            "run_colors": "Matplotlib tab10 categorical palette assigned by initialization seed within each component-count group; palette reused across groups so corresponding seeds share colors.",
            "parameter_change_histograms": "Grouped side-by-side counts in 10 equal bins spanning [-pi, pi]; blue/orange/green encode 2/3/4 target frequencies.",
            "parameter_spider_profiles": "Seven polar spokes; radial coordinate maps wrapped angle [-pi,pi) linearly to [0,1]. Dashed is initial (1.95 pt), solid is final (2.4 pt); radial connectors are 1.95 pt. These line widths are 3x the original values.",
            "per_gate_circular_changes": "For each parameter, draw all shortest signed angular arcs at radius one first, then draw all endpoints above every arc; hollow circles with thick colored outlines are initial and filled squares with dark outlines are final. Zero changes overlap exactly.",
        },
        "outputs": outputs,
    }
    (output_dir / "parameter_plot_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return outputs


def refresh_group_report(output_dir: Path) -> None:
    """Update plot descriptions/links in the existing static group report."""
    report_path = output_dir / "index.html"
    if not report_path.exists():
        return
    html = report_path.read_text(encoding="utf-8")
    html = html.replace(
        "Each histogram uses the shortest signed angular change modulo 2π.",
        "Each histogram uses the shortest signed angular change modulo 2π. Side-by-side bars retain blue (2), orange (3), and green (4 frequency) counts even when they fall in the same bin.",
    )
    html = html.replace(
        "Thin radial connectors join each gate's initial and final positions.",
        "Thin radial connectors join each gate's initial and final positions. Colors are distinct within each frequency-count group and reused across groups by initialization seed.",
    )
    html = html.replace(
        "Each colored arc connects the initial angle marker (circle) to the final marker (square) along the shortest circular path.",
        "Each colored arc connects the initial angle marker (hollow circle) to the final marker (filled square) along the shortest circular path. All connectors are drawn first, then all markers are drawn above them with contrasting outlines. An unchanged angle has overlapping endpoints and no visible arc.",
    )
    html = html.replace(
        "The connector is drawn above the polar grid and markers are drawn last.",
        "All connectors are drawn first, then all markers are drawn above them with contrasting outlines.",
    )
    html = html.replace(
        "Run colors map to IDs in summary.json and metrics.csv.",
        "Colors are distinct among runs within each frequency-count panel and reused across panels by initialization seed; run IDs and hex colors are in summary.json and metrics.csv.",
    )
    if "parameter_plot_manifest.json" not in html:
        html = html.replace(
            '<a href="summary.json">Machine-readable summary</a>',
            '<a href="summary.json">Machine-readable summary</a> · <a href="parameter_plot_manifest.json">Plot methods and manifest</a>',
        )
    if "refresh_component_sweep_visualizations.py" not in html:
        html += ('\n<p>Regenerate these plots from saved results with '
                 '<code>python scripts/refresh_component_sweep_visualizations.py</code>; no training is performed. '
                 'Plot code: <code>src/qftlab/visualization/parameter_summaries.py</code>.</p>\n')
    report_path.write_text(html, encoding="utf-8")
