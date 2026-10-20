"""Convert a binary silhouette into a Fourier-driven CoreXY simulation."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import solve_ivp

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corexy.model import CoreXYModel
from inverse_dynamics import inverse_dynamics, trajectory_state
from plot_style import FIGURE_SIZE, SQUARE_FIGURE_SIZE, configure_plots, format_axes, save_figure

from fourier_curve import (
    FourierCurve,
    load_silhouette,
    ordered_contour,
    resample_by_arclength,
    scale_to_workspace,
)


def save_processing_plots(
    gray: np.ndarray,
    mask: np.ndarray,
    contour: np.ndarray,
    parameter: np.ndarray,
    sampled: np.ndarray,
    reconstructed: np.ndarray,
    curve: FourierCurve,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=SQUARE_FIGURE_SIZE, constrained_layout=True)
    ax.imshow(gray, cmap="gray")
    ax.set_axis_off()
    save_figure(fig, output / "01_input_image.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=SQUARE_FIGURE_SIZE, constrained_layout=True)
    ax.imshow(mask, cmap="gray")
    ax.set_axis_off()
    save_figure(fig, output / "02_silhouette.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=SQUARE_FIGURE_SIZE, constrained_layout=True)
    ax.plot(contour[:, 0], contour[:, 1])
    ax.scatter(contour[0, 0], contour[0, 1], color="red", s=45, label=r"$\mathrm{Start}$", zorder=2)
    ax.set(xlabel=r"$x\ [\mathrm{pixel}]$", ylabel=r"$y\ [\mathrm{pixel}]$")
    format_axes(ax, equal=True)
    ax.legend()
    save_figure(fig, output / "03_ordered_contour.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=SQUARE_FIGURE_SIZE, constrained_layout=True)
    color = ax.scatter(sampled[:, 0], sampled[:, 1], c=parameter, s=12, cmap="viridis")
    ax.set(xlabel=r"$x\ [\mathrm{m}]$", ylabel=r"$y\ [\mathrm{m}]$")
    format_axes(ax, equal=True)
    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes("right", size="5%", pad=0.15)
    colorbar = fig.colorbar(color, cax=colorbar_ax)
    colorbar.set_label(r"$\mathrm{Normalized\ cumulative\ arc\ length}$")
    colorbar.locator = matplotlib.ticker.MaxNLocator(nbins=5)
    colorbar.update_ticks()
    colorbar.ax.tick_params(width=1.6, length=7)
    colorbar.outline.set_linewidth(1.6)
    save_figure(fig, output / "04_arclength_samples.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    ax.plot(parameter, sampled[:, 0], label=r"$x(s)$")
    ax.plot(parameter, sampled[:, 1], label=r"$y(s)$")
    ax.set(xlabel=r"$s$", ylabel=r"$\mathrm{Position}\ [\mathrm{m}]$")
    format_axes(ax)
    ax.legend()
    save_figure(fig, output / "05_arclength_coordinates.png")
    plt.close(fig)

    order = np.argsort(curve.frequencies)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    markerline, stemlines, baseline = ax.stem(curve.frequencies[order], np.abs(curve.coefficients[order]))
    plt.setp(stemlines, linewidth=1.8)
    plt.setp(markerline, markersize=5)
    plt.setp(baseline, linewidth=1.6)
    ax.set(xlabel=r"$k$", ylabel=r"$|c_k|\ [\mathrm{m}]$")
    format_axes(ax)
    save_figure(fig, output / "06_fourier_spectrum.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=SQUARE_FIGURE_SIZE, constrained_layout=True)
    ax.plot(sampled[:, 0], sampled[:, 1], label=r"$\mathrm{Arc\mbox{-}length\ contour}$", alpha=0.75)
    ax.plot(reconstructed[:, 0], reconstructed[:, 1], "--", label=r"$\mathrm{Fourier\ reconstruction}$")
    ax.set(xlabel=r"$x\ [\mathrm{m}]$", ylabel=r"$y\ [\mathrm{m}]$")
    format_axes(ax, equal=True)
    ax.legend()
    save_figure(fig, output / "07_reconstruction.png")
    plt.close(fig)


def simulate(curve: FourierCurve, period: float, samples: int):
    model = CoreXYModel()
    time = np.linspace(0.0, period, samples)

    def prescribed_voltage(t: float) -> np.ndarray:
        _, velocity, acceleration, jerk = curve.time_derivatives(t, period)
        return inverse_dynamics(model, velocity[0], acceleration[0], jerk[0])

    position_0, velocity_0, acceleration_0, _ = curve.time_derivatives(0.0, period)
    state_0 = trajectory_state(model, position_0[0], velocity_0[0], acceleration_0[0])
    solution = solve_ivp(
        lambda t, state: model.derivative(t, state, prescribed_voltage(t)),
        (0.0, period), state_0, t_eval=time, rtol=1e-9, atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    reference = curve.time_derivatives(time, period)[0].T
    actual = solution.y[[0, 3]]
    voltage = np.column_stack([prescribed_voltage(t) for t in time])
    return time, solution.y, reference, actual, voltage


def run(image: Path, order: int, contour_samples: int, simulation_samples: int,
        workspace_size: float, period: float, output_root: Path) -> Path:
    configure_plots()
    output = output_root / image.stem
    output.mkdir(parents=True, exist_ok=True)

    gray, mask = load_silhouette(image)
    contour = ordered_contour(mask)
    parameter, sampled_pixels = resample_by_arclength(contour, contour_samples)
    sampled = scale_to_workspace(sampled_pixels, workspace_size)
    curve = FourierCurve.fit(sampled, order)
    reconstructed = curve.evaluate(parameter)

    np.savetxt(output / "ordered_contour_pixels.csv", contour, delimiter=",", header="x,y", comments="")
    np.savetxt(output / "arclength_contour_m.csv", np.column_stack((parameter, sampled)),
               delimiter=",", header="s,x,y", comments="")
    curve.save(output / "fourier_coefficients.csv")
    np.savetxt(output / "reconstruction_m.csv", np.column_stack((parameter, reconstructed)),
               delimiter=",", header="s,x,y", comments="")
    save_processing_plots(gray, mask, contour, parameter, sampled, reconstructed, curve, output)

    time, state, reference, actual, voltage = simulate(curve, period, simulation_samples)
    np.savez(output / "simulation.npz", time=time, state=state,
             reference=reference, actual=actual, voltage=voltage)

    fig, ax = plt.subplots(figsize=SQUARE_FIGURE_SIZE, constrained_layout=True)
    ax.plot(*reference, "--", label=r"$\mathrm{Fourier\ trajectory}$")
    ax.plot(*actual, label=r"$\mathrm{Simulated\ dynamics}$")
    ax.set(xlabel=r"$x\ [\mathrm{m}]$", ylabel=r"$y\ [\mathrm{m}]$")
    format_axes(ax, equal=True)
    ax.legend()
    save_figure(fig, output / "08_trajectory.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    ax.plot(time, voltage[0], label=r"$u_x$")
    ax.plot(time, voltage[1], label=r"$u_y$")
    ax.set(xlabel=r"$t\ [\mathrm{s}]$", ylabel=r"$\mathrm{Voltage}\ [\mathrm{V}]$")
    format_axes(ax)
    ax.legend()
    save_figure(fig, output / "09_voltages.png")
    plt.close(fig)

    error = np.linalg.norm(actual - reference, axis=0)
    summary = (
        f"input={image}\norder={order}\ncontour_samples={contour_samples}\n"
        f"workspace_size_m={workspace_size}\nperiod_s={period}\n"
        f"maximum_tracking_error_m={error.max():.6e}\n"
        f"rms_tracking_error_m={np.sqrt(np.mean(error**2)):.6e}\n"
        f"maximum_absolute_voltage_V={np.abs(voltage).max():.6e}\n"
    )
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="?", default=BASE_DIR / "inputs" / "Catalunya.png")
    parser.add_argument("--order", type=int, default=60)
    parser.add_argument("--contour-samples", type=int, default=2048)
    parser.add_argument("--simulation-samples", type=int, default=2001)
    parser.add_argument("--workspace-size", type=float, default=0.16, help="largest shape span [m]")
    parser.add_argument("--period", type=float, default=8.0, help="drawing period [s]")
    parser.add_argument("--output-root", type=Path, default=BASE_DIR / "outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    destination = run(args.image, args.order, args.contour_samples,
                      args.simulation_samples, args.workspace_size,
                      args.period, args.output_root)
    print(f"Results saved in {destination}")

