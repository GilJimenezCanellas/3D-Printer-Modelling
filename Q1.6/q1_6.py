from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipe

from filament_model import FilamentCoolingModel
from plot_style import FIGURE_SIZE, configure_plots, format_axes, save_figure


A = 0.08
B = 0.04
PRINT_SPEED = 0.04
UNIFORMITY_TOLERANCE = 1.0


def main() -> None:
    configure_plots()
    length = 4.0 * A * ellipe(1.0 - (B / A) ** 2)
    model = FilamentCoolingModel(length, PRINT_SPEED)
    solution = model.simulate(UNIFORMITY_TOLERANCE)
    output_directory = Path("outputs")
    output_directory.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    image = ax.pcolormesh(
        solution.arc_length,
        solution.time,
        solution.temperature,
        shading="nearest",
        cmap="turbo",
    )
    ax.axhline(solution.print_time, color="black", linestyle="--", label=r"$t_{\mathrm{print}}$")
    ax.set(xlabel=r"$s\ [\mathrm{m}]$", ylabel=r"$t\ [\mathrm{s}]$")
    format_axes(ax)
    ax.legend()
    fig.colorbar(image, ax=ax, label=r"$T\ [^{\circ}\mathrm{C}]$")
    heatmap_path = output_directory / "q1_6_temperature_field.png"
    save_figure(fig, heatmap_path)
    plt.close(fig)

    minimum = np.nanmin(solution.temperature, axis=1)
    maximum = np.nanmax(solution.temperature, axis=1)
    mean = np.nanmean(solution.temperature, axis=1)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    ax.plot(solution.time, maximum, label=r"$T_{\max}$")
    ax.plot(solution.time, mean, label=r"$\overline{T}$")
    ax.plot(solution.time, minimum, label=r"$T_{\min}$")
    ax.axvline(solution.print_time, color="black", linestyle="--", label=r"$t_{\mathrm{print}}$")
    ax.axhline(model.p.equilibrium_temperature, color="gray", linestyle=":", label=r"$T_{\mathrm{eq}}$")
    ax.set(xlabel=r"$t\ [\mathrm{s}]$", ylabel=r"$T\ [^{\circ}\mathrm{C}]$")
    format_axes(ax)
    ax.legend(ncol=2)
    history_path = output_directory / "q1_6_temperature_history.png"
    save_figure(fig, history_path)
    plt.close(fig)

    final_range = np.ptp(solution.temperature[-1])
    results = (
        f"Ellipse perimeter: {length:.4f} m\n"
        f"Printing time: {solution.print_time:.2f} s\n"
        f"Nearly uniform at: {solution.uniform_time:.2f} s\n"
        f"Cooling after printing: {solution.uniform_time - solution.print_time:.2f} s\n"
        f"Final temperature range: {final_range:.3f} degC\n"
        f"Asymptotic equilibrium temperature: {model.p.equilibrium_temperature:.2f} degC\n"
    )
    results_path = output_directory / "q1_6_results.txt"
    results_path.write_text(results)
    print(results, end="")
    print(f"Saved {heatmap_path}")
    print(f"Saved {history_path}")
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
