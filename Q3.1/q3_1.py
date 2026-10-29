"""Q3.1: Spain print with filament-empty and colour-change events."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
sys.path[:0] = [str(ROOT), str(BASE_DIR)]

from hybrid_model import HybridPrinter, MODE_NAMES, Mode, Scenario, load_fourier_curve
from plot_style import configure_plots, format_axes, save_figure


def print_sections(solution) -> list[np.ndarray]:
    indices = np.flatnonzero(solution.mode == int(Mode.PRINT))
    return list(np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1))


def mark_modes(axes, solution) -> None:
    boundaries = [0.0] + [item.time for item in solution.transitions]
    modes = [Mode(int(solution.mode[0]))] + [
        item.target for item in solution.transitions[:-1]
    ]
    colors = plt.colormaps["tab10"](np.linspace(0.0, 1.0, 7))
    first_wipe = True
    for left, right, mode in zip(boundaries[:-1], boundaries[1:], modes):
        for ax in axes:
            ax.axvline(right, color="0.45", linewidth=0.9, alpha=0.65)
        axes[0].axvspan(left, right, color=colors[int(mode)], alpha=0.08)
        label_time = 0.5 * (left + right)
        if mode == Mode.WIPE and first_wipe:
            label_time += 0.7
            first_wipe = False
        axes[0].text(
            label_time,
            1.01,
            f"S{int(mode)}",
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=12,
        )
    print_events = [
        item for item in solution.transitions
        if item.source == Mode.PRINT and item.target == Mode.RETURN_HOME
    ]
    event_labels = [
        r"$S0\rightarrow S1:\ \mathrm{Filament\ empty}$",
        r"$S0 \rightarrow S1$: “C” Triggered",
    ]
    for index, (item, label) in enumerate(zip(print_events, event_labels)):
        axes[-1].plot(
            [item.time, item.time],
            [-0.13, 0.0],
            transform=axes[-1].get_xaxis_transform(),
            color="0.25",
            linewidth=1.4,
            clip_on=False,
        )
        axes[-1].annotate(
            label,
            xy=(item.time, 0.0),
            xycoords=("data", "axes fraction"),
            xytext=((85 if index == 1 else 0), -42),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold",
        )


def plot_histories(model, solution, destination: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 11.5), sharex=True, constrained_layout=True)
    time = solution.time

    axes[0].plot(time, solution.temperature, label=r"$T_n$")
    axes[0].axhline(210.0, color="black", linestyle="--", label=r"$T_{\mathrm{set}}$")
    axes[0].set(ylabel=r"$T_n\ [^\circ\mathrm{C}]$")
    axes[0].legend(ncol=2)

    axes[1].plot(time, solution.x, label=r"$x$")
    axes[1].plot(time, solution.saved_x, "--", label=r"$x_s$")
    axes[1].plot(time, solution.y, label=r"$y$")
    axes[1].plot(time, solution.saved_y, "--", label=r"$y_s$")
    axes[1].set(ylabel=r"$\mathrm{Position}\ [\mathrm{m}]$")
    axes[1].legend(ncol=4)

    axes[2].plot(time, solution.tau)
    axes[2].axhline(
        model.p.tau_1,
        color="black",
        linestyle="--",
        label=r"$\tau_1$",
    )
    axes[2].axhline(
        model.p.tau_3,
        color="tab:red",
        linestyle="-.",
        label=r"$\tau_3$",
    )
    axes[2].axhline(
        model.p.tau_2,
        color="0.4",
        linestyle=":",
        label=r"$\tau_2$",
    )
    axes[2].set(xlabel=r"$t\ [\mathrm{s}]$", ylabel=r"$\tau\ [\mathrm{s}]$")
    axes[2].legend(
        ncol=3,
        loc="center",
        bbox_to_anchor=(0.75, 9.0),
        bbox_transform=axes[2].get_yaxis_transform(),
    )

    mark_modes(axes, solution)
    for ax in axes:
        format_axes(ax)
    save_figure(fig, destination)
    plt.close(fig)


def plot_print_stages(model, solution, output: Path) -> None:
    sections = print_sections(solution)
    reference = model.curve.evaluate(np.linspace(0.0, 1.0, 2000))
    colors = ["tab:blue", "tab:blue", "tab:orange"]
    for stage in range(3):
        fig, ax = plt.subplots(figsize=(6.8, 6.8), constrained_layout=True)
        ax.plot(
            reference[:, 0],
            reference[:, 1],
            "--",
            color="0.65",
            linewidth=1.5,
            label=r"$\mathrm{Spain\ reference}$",
        )
        for index, section in enumerate(sections[: stage + 1]):
            label = (
                r"$\mathrm{Original\ colour}$"
                if index == 0
                else "_nolegend_"
                if index == 1
                else r"$\mathrm{New\ colour}$"
            )
            ax.plot(
                solution.x[section],
                solution.y[section],
                color=colors[index],
                label=label,
            )
        empty_point = sections[0][-1]
        ax.scatter(
            solution.x[empty_point],
            solution.y[empty_point],
            s=80,
            color="black",
            zorder=5,
            label=r"$\mathrm{Filament\ empty}$",
        )
        if stage >= 1:
            saved = sections[1][-1]
            ax.scatter(
                solution.x[saved],
                solution.y[saved],
                s=90,
                color="red",
                zorder=4,
                label=r"“C” $\mathrm{Triggered}$",
            )
        ax.set(
            xlabel=r"$x\ [\mathrm{m}]$",
            ylabel=r"$y\ [\mathrm{m}]$",
        )
        format_axes(ax, equal=True)
        ax.set(xlim=(0.0, None), ylim=(0.0, None))
        ax.legend(fontsize=14)
        save_figure(fig, output / f"q3_1_spain_stage_{stage + 1}.png")
        plt.close(fig)


def main() -> None:
    configure_plots()
    curve_path = ROOT / "random-shape" / "outputs" / "España" / "fourier_coefficients.csv"
    model = HybridPrinter(load_fourier_curve(curve_path), print_time=36.0)
    scenario = Scenario((("filament", 1.0 / 3.0), ("colour", 0.5)))
    solution = model.simulate(scenario)
    output = BASE_DIR / "outputs"
    output.mkdir(exist_ok=True)

    np.savez(
        output / "combined_replacements.npz",
        time=solution.time,
        state=solution.state,
        mode=solution.mode,
        reference=solution.reference,
    )
    transition_text = "\n".join(
        f"{item.time:9.3f} s: S{int(item.source)} -> "
        f"{('end' if item.target is None else f'S{int(item.target)}')} "
        f"[{item.guard}; {item.reset}]"
        for item in solution.transitions
    )
    (output / "combined_replacements_transitions.txt").write_text(
        transition_text + "\n", encoding="utf-8"
    )
    plot_histories(model, solution, output / "q3_1_hybrid_histories.png")
    plot_print_stages(model, solution, output)

    print(f"Simulation time: {solution.time[-1]:.2f} s")
    for item in solution.transitions:
        target = "end" if item.target is None else f"S{int(item.target)}"
        print(
            f"{item.time:8.2f} s  S{int(item.source)} "
            f"({MODE_NAMES[item.source]}) -> {target}: {item.guard}"
        )


if __name__ == "__main__":
    main()
