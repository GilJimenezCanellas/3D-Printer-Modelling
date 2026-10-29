"""Q3.2: draw a perfect square using analytical feedforward voltages."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
RANDOM_SHAPE_DIR = ROOT / "random-shape"
sys.path[:0] = [str(ROOT), str(RANDOM_SHAPE_DIR)]

from corexy.model import CoreXYModel
from inverse_dynamics import inverse_dynamics
from plot_style import configure_plots, format_axes, save_figure


Vector = NDArray[np.float64]
SIDE_LENGTH = 0.10  # L [m]
SEGMENT_DURATION = 1.0  # T_s [s]
SAMPLES_PER_SEGMENT = 301

# Counter-clockwise traversal from (0.05, 0.05): bottom, right, top, left.
SQUARE_MIN = 0.05  # [m]
SQUARE_MAX = SQUARE_MIN + SIDE_LENGTH
CORNERS = np.array([
    [SQUARE_MIN, SQUARE_MIN],
    [SQUARE_MAX, SQUARE_MIN],
    [SQUARE_MAX, SQUARE_MAX],
    [SQUARE_MIN, SQUARE_MAX],
    [SQUARE_MIN, SQUARE_MIN],
])


def quintic_profile(local_time: float | Vector) -> tuple[Vector, Vector, Vector, Vector]:
    """Return s and its first three time derivatives on one edge."""
    q = np.asarray(local_time, dtype=float) / SEGMENT_DURATION
    s = 10.0 * q**3 - 15.0 * q**4 + 6.0 * q**5
    s_dot = (30.0 * q**2 - 60.0 * q**3 + 30.0 * q**4) / SEGMENT_DURATION
    s_ddot = (60.0 * q - 180.0 * q**2 + 120.0 * q**3) / SEGMENT_DURATION**2
    s_jerk = (60.0 - 360.0 * q + 360.0 * q**2) / SEGMENT_DURATION**3
    return s, s_dot, s_ddot, s_jerk


def edge_kinematics(edge: int, local_time: float | Vector) -> tuple[Vector, Vector, Vector, Vector]:
    """Return position, velocity, acceleration, and jerk for one square edge."""
    s, s_dot, s_ddot, s_jerk = quintic_profile(local_time)
    start = CORNERS[edge]
    displacement = CORNERS[edge + 1] - start
    position = start[:, None] + displacement[:, None] * np.atleast_1d(s)
    velocity = displacement[:, None] * np.atleast_1d(s_dot)
    acceleration = displacement[:, None] * np.atleast_1d(s_ddot)
    jerk = displacement[:, None] * np.atleast_1d(s_jerk)
    if np.ndim(local_time) == 0:
        return position[:, 0], velocity[:, 0], acceleration[:, 0], jerk[:, 0]
    return position, velocity, acceleration, jerk


def simulate_square(model: CoreXYModel) -> tuple[Vector, Vector, Vector, Vector]:
    """Simulate all four edges, stopping at every corner by construction."""
    time_parts: list[Vector] = []
    state_parts: list[NDArray[np.float64]] = []
    voltage_parts: list[NDArray[np.float64]] = []
    reference_parts: list[NDArray[np.float64]] = []
    state = np.array([CORNERS[0, 0], 0.0, 0.0, CORNERS[0, 1], 0.0, 0.0])

    for edge in range(4):
        local_time = np.linspace(0.0, SEGMENT_DURATION, SAMPLES_PER_SEGMENT)

        def voltage_at(time: float) -> Vector:
            _, velocity, acceleration, jerk = edge_kinematics(edge, time)
            return inverse_dynamics(model, velocity, acceleration, jerk)

        result = solve_ivp(
            lambda time, current: model.derivative(time, current, voltage_at(time)),
            (0.0, SEGMENT_DURATION),
            state,
            t_eval=local_time,
            rtol=1e-10,
            atol=1e-12,
        )
        if not result.success:
            raise RuntimeError(result.message)

        reference, _, _, _ = edge_kinematics(edge, local_time)
        voltages = np.column_stack([voltage_at(time) for time in local_time])
        start = 0 if edge == 0 else 1
        time_parts.append(edge * SEGMENT_DURATION + local_time[start:])
        state_parts.append(result.y[:, start:])
        voltage_parts.append(voltages[:, start:])
        reference_parts.append(reference[:, start:])
        state = result.y[:, -1]

    return (
        np.concatenate(time_parts),
        np.hstack(state_parts),
        np.hstack(voltage_parts),
        np.hstack(reference_parts),
    )


def plot_square(position: Vector, reference: Vector, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    ax.plot(reference[0], reference[1], "--", color="0.45", label=r"$\mathrm{Reference}$")
    ax.plot(position[0], position[1], label=r"$\mathrm{Feedforward\ response}$")
    ax.scatter(CORNERS[:-1, 0], CORNERS[:-1, 1], color="black", s=35, zorder=4)
    ax.set(xlabel=r"$x\ [\mathrm{m}]$", ylabel=r"$y\ [\mathrm{m}]$")
    plot_limit = SQUARE_MAX + 0.01
    ax.set(xlim=(0.0, plot_limit), ylim=(0.0, plot_limit))
    format_axes(ax, equal=True)
    ax.legend()
    save_figure(fig, destination)
    plt.close(fig)


def plot_histories(time: Vector, position: Vector, voltage: Vector, destination: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True, constrained_layout=True)
    axes[0].plot(time, voltage[0], label=r"$u_x$")
    axes[0].plot(time, voltage[1], label=r"$u_y$")
    axes[0].set(ylabel=r"$\mathrm{Voltage}\ [\mathrm{V}]$")
    axes[0].legend(ncol=2)

    axes[1].plot(time, position[0], label=r"$x$")
    axes[1].plot(time, position[1], label=r"$y$")
    axes[1].set(
        xlabel=r"$t\ [\mathrm{s}]$",
        ylabel=r"$\mathrm{Position}\ [\mathrm{m}]$",
        xlim=(0.0, 4.0 * SEGMENT_DURATION),
        ylim=(0.0, SQUARE_MAX + 0.01),
    )
    axes[1].legend(ncol=2)

    for edge_end in np.arange(1, 4) * SEGMENT_DURATION:
        for ax in axes:
            ax.axvline(edge_end, color="0.65", linestyle=":", linewidth=1.2)
    for ax in axes:
        format_axes(ax)
    save_figure(fig, destination)
    plt.close(fig)


def main() -> None:
    configure_plots()
    model = CoreXYModel()
    time, state, voltage, reference = simulate_square(model)
    position = state[[0, 3]]

    output = BASE_DIR / "outputs"
    output.mkdir(exist_ok=True)
    square_path = output / "q3_2_perfect_square.png"
    histories_path = output / "q3_2_square_histories.png"
    plot_square(position, reference, square_path)
    plot_histories(time, position, voltage, histories_path)

    maximum_error = np.max(np.linalg.norm(position - reference, axis=0))
    print(f"Saved {square_path}")
    print(f"Saved {histories_path}")
    print(f"Maximum position error: {maximum_error:.3e} m")


if __name__ == "__main__":
    main()
