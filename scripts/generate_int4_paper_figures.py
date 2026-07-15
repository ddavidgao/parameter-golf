#!/usr/bin/env python3
"""Generate the primary figures for the W4 value-differencing paper."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
TRAJECTORY_CSV = PAPER / "seed42_paired_compression_trajectory.csv"
CONTROLS_CSV = PAPER / "data" / "w4_controls_adjudicated.csv"
SYMMETRIC_LOG = PAPER / "data" / "w4_symmetric_quantize_both.log"

STANDARD = "#4C78A8"
DG = "#E45756"
GRID = "#D9DEE7"
TEXT = "#20242A"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.2,
            "axes.edgecolor": "#9AA1AB",
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "scripts/generate_int4_paper_figures.py"}
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight", metadata=metadata)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def trajectory_figure() -> None:
    rows = []
    with TRAJECTORY_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["state"] == "uniform_int4":
                rows.append(row)

    steps = [int(row["step"]) for row in rows]
    gaps = [float(row["standard_minus_dg_damage_bpb"]) for row in rows]

    fig, ax = plt.subplots(figsize=(6.25, 3.25), constrained_layout=True)
    ax.axhline(0, color="#747B85", linewidth=0.9)
    ax.axvspan(17_000, 20_000, color="#EEF0F3", zorder=0)
    ax.plot(
        steps,
        gaps,
        color=DG,
        marker="o",
        markersize=4.5,
        linewidth=2.0,
        label="Standard damage minus value-difference damage",
        zorder=3,
    )
    ax.fill_between(steps, 0, gaps, color=DG, alpha=0.10, zorder=1)

    ax.text(18_500, 0.0018, "LR decay", ha="center", va="bottom", color="#5D646E")
    ax.annotate(
        f"+{gaps[-1]:.4f} BPB",
        xy=(steps[-1], gaps[-1]),
        xytext=(-8, 9),
        textcoords="offset points",
        ha="right",
        color=DG,
        fontweight="bold",
    )

    ax.set_title("W4 damage gap through training (seed 42)", loc="left")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Damage gap (BPB): standard - value difference")
    ax.set_xlim(1_500, 20_500)
    ax.set_ylim(-0.001, max(gaps) * 1.20)
    ax.set_xticks([2_000, 4_000, 6_000, 8_000, 10_000, 12_000, 14_000, 16_000, 18_000, 20_000])
    ax.set_xticklabels(["2K", "4K", "6K", "8K", "10K", "12K", "14K", "16K", "18K", "20K"])
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "int4_damage_gap_trajectory")


def mechanism_figure() -> None:
    fig, (ax_flow, ax_depth) = plt.subplots(
        1,
        2,
        figsize=(6.25, 2.35),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
        constrained_layout=True,
    )

    ax_flow.set_xlim(0, 12)
    ax_flow.set_ylim(0, 6)
    ax_flow.axis("off")
    ax_flow.set_title("(a) Value path", loc="left")

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        color: str,
        lw: float = 1.0,
        fontsize: float = 8.6,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor=color,
            edgecolor="#68717D",
            linewidth=lw,
        )
        ax_flow.add_patch(patch)
        ax_flow.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize)

    def arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#68717D") -> None:
        ax_flow.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.1,
                color=color,
            )
        )

    ax_flow.text(0.25, 5.02, "Previous", color="#5D646E", fontsize=7.3, fontweight="bold")
    ax_flow.text(0.25, 1.95, "Current", color="#5D646E", fontsize=7.3, fontweight="bold")
    box(0.25, 4.05, 1.35, 0.72, r"$x_{t-1}$", "#F2F4F7")
    box(0.25, 0.98, 1.35, 0.72, r"$x_t$", "#F2F4F7")
    box(2.15, 1.05, 1.85, 3.65, "Shared\nvalue\nprojection\n" + r"$W_V^{(\ell)}$", "#E9F0F8", fontsize=7.2)
    box(4.55, 4.05, 1.25, 0.72, r"$p_{t-1}$", "#F2F6FA")
    box(4.55, 0.98, 1.25, 0.72, r"$p_t$", "#F2F6FA")
    arrow(1.6, 4.41, 2.15, 4.41)
    arrow(1.6, 1.34, 2.15, 1.34)
    arrow(4.0, 4.41, 4.55, 4.41)
    arrow(4.0, 1.34, 4.55, 1.34)

    merge = Circle((6.75, 2.88), 0.42, facecolor="#FDECEA", edgecolor="#68717D", linewidth=1.0)
    ax_flow.add_patch(merge)
    ax_flow.text(6.75, 2.88, r"$\Sigma$", ha="center", va="center", fontsize=9.0)
    arrow(5.80, 4.41, 6.48, 3.20, DG)
    arrow(5.80, 1.34, 6.48, 2.56, DG)
    ax_flow.text(6.05, 3.93, r"$-\alpha_\ell$", color=DG, fontsize=7.5, ha="center")
    ax_flow.text(6.05, 1.83, r"$+1$", color=DG, fontsize=7.5, ha="center")
    box(7.65, 2.48, 1.25, 0.8, r"$\widetilde p_t$", "#FDECEA", fontsize=8.4)
    arrow(7.17, 2.88, 7.65, 2.88, DG)
    box(10.0, 2.48, 1.35, 0.8, "SDPA", "#EEF4EC", fontsize=8.0)
    arrow(8.90, 2.88, 10.0, 2.88, DG)
    arrow(10.68, 4.55, 10.68, 3.28)
    ax_flow.text(10.68, 4.90, r"$Q_\ell, K_\ell$", ha="center", color="#5D646E", fontsize=7.4)

    ax_depth.set_xlim(0, 10)
    ax_depth.set_ylim(0, 6)
    ax_depth.axis("off")
    ax_depth.set_title("(b) Depth schedule", loc="left")

    def depth_card(y: float, layer: int, alpha: str, equation: str, label: str, fill: str) -> None:
        card = FancyBboxPatch(
            (1.0, y),
            8.7,
            1.2,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor=fill,
            edgecolor="#B7BEC8",
            linewidth=0.9,
        )
        ax_depth.add_patch(card)
        ax_depth.add_patch(
            FancyBboxPatch(
                (1.0, y),
                0.20,
                1.2,
                boxstyle="round,pad=0.0,rounding_size=0.06",
                facecolor=DG if layer else "#9AA1AB",
                edgecolor="none",
            )
        )
        ax_depth.text(1.48, y + 0.81, f"Layer {layer}  ·  α = {alpha}", fontsize=7.5, fontweight="bold", color=TEXT)
        ax_depth.text(1.48, y + 0.34, equation, fontsize=8.0, color=TEXT)

    depth_card(4.25, 0, "0", r"$\widetilde p_t=p_t$", "", "#F4F5F7")
    depth_card(2.45, 5, "0.5", r"$\widetilde p_t=p_t-0.5p_{t-1}$", "", "#FFF4F3")
    depth_card(0.65, 10, "1", r"$\widetilde p_t=p_t-p_{t-1}$", "", "#FDE8E6")
    ax_depth.add_patch(
        FancyArrowPatch(
            (0.45, 5.25),
            (0.45, 0.70),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color="#7C838D",
        )
    )
    ax_depth.text(0.13, 3.0, "deeper", rotation=90, ha="center", va="center", color="#5D646E", fontsize=7.2)
    save_figure(fig, "value_difference_mechanism")


def endpoint_reversal_figure() -> None:
    values = {
        1337: {
            "standard": (1.1539035, 1.2405266),
            "value difference": (1.1605306, 1.2307418),
        },
        42: {
            "standard": (1.1571615, 1.2482915),
            "value difference": (1.1596927, 1.2304696),
        },
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.45), sharex=True, sharey=True, constrained_layout=True)
    conditions = [("Full precision", 0), ("After W4", 1)]
    for ax, seed in zip(axes, [1337, 42], strict=True):
        for condition, index in conditions:
            y = 1 - index
            ax.plot(
                values[seed]["standard"][index],
                y,
                marker="o",
                markersize=6,
                color=STANDARD,
                linestyle="none",
                label="Standard" if index == 0 else None,
                zorder=3,
            )
            ax.plot(
                values[seed]["value difference"][index],
                y,
                marker="D",
                markersize=5.5,
                color=DG,
                linestyle="none",
                label="Value difference" if index == 0 else None,
                zorder=3,
            )
            ax.annotate(
                f"{values[seed]['standard'][index]:.3f}",
                xy=(values[seed]["standard"][index], y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                color=STANDARD,
                fontsize=7.5,
            )
            ax.annotate(
                f"{values[seed]['value difference'][index]:.3f}",
                xy=(values[seed]["value difference"][index], y),
                xytext=(0, -11),
                textcoords="offset points",
                ha="center",
                color=DG,
                fontsize=7.5,
            )
        ax.set_title(f"Seed {seed}")
        ax.set_xlim(1.145, 1.255)
        ax.set_yticks([1, 0], ["Full precision", "After W4"])
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Validation BPB (lower is better)")
    axes[0].legend(loc="center right", frameon=False)
    fig.suptitle("The lower-BPB model changes after W4", fontsize=10.5, fontweight="bold")
    save_figure(fig, "int4_endpoint_reversal")


def load_adjudicated_damage() -> dict[tuple[int, str, str], float]:
    data: dict[tuple[int, str, str], float] = {}
    with CONTROLS_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            data[(int(row["seed"]), row["architecture"], row["state"])] = float(row["damage_bpb"])
    return data


def load_symmetric_attention_damage() -> dict[tuple[int, str], float]:
    fp: dict[tuple[int, str], float] = {}
    quantized: dict[tuple[int, str], float] = {}
    pattern = re.compile(
        r"^(?:std|dg)_seed(?P<seed>\d+),(?P<arch>standard|dg),(?P<state>fp|isolated_int4_attn),normal,[^,]+,(?P<bpb>[0-9.]+)$"
    )
    for line in SYMMETRIC_LOG.read_text().splitlines():
        match = pattern.match(line)
        if not match:
            continue
        key = (int(match.group("seed")), match.group("arch"))
        bpb = float(match.group("bpb"))
        if match.group("state") == "fp":
            fp[key] = bpb
        else:
            quantized[key] = bpb
    return {key: quantized[key] - fp[key] for key in fp}


def localization_figure() -> None:
    adjudicated = load_adjudicated_damage()
    attention = load_symmetric_attention_damage()
    module_states = [
        ("Attention", None),
        ("MLP", "isolated_int4_mlp"),
    ]
    projection_states = [
        ("Value", "isolated_int4_attn_payload"),
        ("Query", "isolated_int4_attn_q"),
        ("Key", "isolated_int4_attn_k"),
        ("Output", "isolated_int4_attn_out"),
    ]

    def calculate_gaps(states: list[tuple[str, str | None]]) -> dict[int, list[float]]:
        gaps: dict[int, list[float]] = {1337: [], 42: []}
        for seed in gaps:
            for _label, state in states:
                if state is None:
                    std_damage = attention[(seed, "standard")]
                    dg_damage = attention[(seed, "dg")]
                else:
                    std_damage = adjudicated[(seed, "standard", state)]
                    dg_damage = adjudicated[(seed, "dg", state)]
                gaps[seed].append(std_damage - dg_damage)
        return gaps

    module_gaps = calculate_gaps(module_states)
    projection_gaps = calculate_gaps(projection_states)
    offset = 0.10

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.25, 2.35),
        gridspec_kw={"width_ratios": [0.85, 1.35]},
        constrained_layout=True,
    )

    def draw_panel(
        ax: plt.Axes,
        states: list[tuple[str, str | None]],
        gaps: dict[int, list[float]],
        title: str,
    ) -> None:
        labels = [label for label, _state in states]
        y = list(range(len(labels)))
        ax.axvline(0, color="#747B85", linewidth=0.9, zorder=1)
        ax.scatter(gaps[1337], [value - offset for value in y], s=42, color=STANDARD, marker="o", label="Seed 1337", zorder=3)
        ax.scatter(gaps[42], [value + offset for value in y], s=38, color=DG, marker="D", label="Seed 42", zorder=3)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(-0.0055, 0.0235)
        ax.set_title(title, loc="left", fontsize=9.2)
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)

    draw_panel(axes[0], module_states, module_gaps, "(a) Whole modules")
    draw_panel(axes[1], projection_states, projection_gaps, "(b) Within attention")
    axes[0].set_xlabel("Damage gap (BPB)")
    axes[1].set_xlabel("Damage gap (BPB)")
    axes[0].legend(loc="lower right", fontsize=7.3)
    fig.suptitle(
        "W4 damage reduction localizes to the value projection",
        fontsize=10.5,
        fontweight="bold",
    )
    save_figure(fig, "int4_component_localization")


def gaussian_control_figure() -> None:
    values = {
        1337: {"Standard": 0.02310, "Value difference": 0.00987, "reduction": "57.3% lower"},
        42: {"Standard": 0.02266, "Value difference": 0.00960, "reduction": "57.6% lower"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.25, 1.9), sharex=True, sharey=True, constrained_layout=True)
    for ax, seed in zip(axes, [1337, 42], strict=True):
        y = [1, 0]
        damages = [values[seed]["Standard"], values[seed]["Value difference"]]
        colors = [STANDARD, DG]
        bars = ax.barh(y, damages, color=colors, height=0.48)
        for bar, damage in zip(bars, damages, strict=True):
            ax.text(damage + 0.00045, bar.get_y() + bar.get_height() / 2, f"{damage:.4f}", va="center", fontsize=7.6)
        ax.set_title(f"Seed {seed}  ·  {values[seed]['reduction']}", fontsize=8.8)
        ax.set_yticks(y, ["Standard", "Value difference"])
        ax.set_xlim(0, 0.027)
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Mean BPB damage")
    fig.suptitle("Matched Gaussian noise produces the same ordering", fontsize=10.5, fontweight="bold")
    save_figure(fig, "gaussian_attention_control")


def main() -> None:
    configure_style()
    mechanism_figure()
    endpoint_reversal_figure()
    trajectory_figure()
    localization_figure()
    gaussian_control_figure()


if __name__ == "__main__":
    main()
