"""Convert a binary silhouette into a Fourier-driven CoreXY simulation."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corexy.model import CoreXYModel
from inverse_dynamics import inverse_dynamics, trajectory_state

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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Input image")
    axes[0].set_axis_off()
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Detected silhouette")
    axes[1].set_axis_off()
    fig.savefig(output / "01_silhouette.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot(contour[:, 0], contour[:, 1], linewidth=1)
    ax.scatter(contour[0, 0], contour[0, 1], color="red", label="start", zorder=2)
    ax.set(title="Ordered contour", xlabel="pixel x", ylabel="pixel y", aspect="equal")
    ax.legend()
    fig.savefig(output / "02_ordered_contour.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    color = axes[0].scatter(sampled[:, 0], sampled[:, 1], c=parameter, s=5, cmap="viridis")
    axes[0].set(title="Uniform arc-length samples", xlabel="x [m]", ylabel="y [m]", aspect="equal")
    fig.colorbar(color, ax=axes[0], label="normalized cumulative arc length")
    axes[1].plot(parameter, sampled[:, 0], label="x(s)")
    axes[1].plot(parameter, sampled[:, 1], label="y(s)")
    axes[1].set(title="Arc-length parametrization", xlabel="s", ylabel="position [m]")
    axes[1].legend()
    fig.savefig(output / "03_arclength.png", dpi=180)
    plt.close(fig)

    order = np.argsort(curve.frequencies)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.stem(curve.frequencies[order], np.abs(curve.coefficients[order]))
    ax.set(title="Fourier coefficient magnitudes", xlabel="harmonic k", ylabel="|c_k| [m]")
    fig.savefig(output / "04_fourier_spectrum.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot(sampled[:, 0], sampled[:, 1], label="arc-length contour", alpha=0.7)
    ax.plot(reconstructed[:, 0], reconstructed[:, 1], "--", label="Fourier reconstruction")
    ax.set(title="Fourier reconstruction", xlabel="x [m]", ylabel="y [m]", aspect="equal")
    ax.legend()
    fig.savefig(output / "05_reconstruction.png", dpi=180)
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    axes[0].plot(*reference, "--", label="Fourier trajectory")
    axes[0].plot(*actual, label="simulated dynamics")
    axes[0].set(title="Resulting tool-head trajectory", xlabel="x [m]", ylabel="y [m]", aspect="equal")
    axes[0].legend()
    axes[1].plot(time, voltage[0], label=r"$u_x$")
    axes[1].plot(time, voltage[1], label=r"$u_y$")
    axes[1].set(title="Inverse-dynamics inputs", xlabel="time [s]", ylabel="voltage [V]")
    axes[1].legend()
    fig.savefig(output / "06_dynamics_and_voltages.png", dpi=180)
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
    parser.add_argument("--order", type=int, default=50)
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

