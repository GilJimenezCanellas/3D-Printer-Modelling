"""Q1.4: propagate the decoupled model using analytical ellipse inputs."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from corexy.model import CoreXYModel
from plot_style import FIGURE_SIZE, SQUARE_FIGURE_SIZE, configure_plots, format_axes, save_figure


# Desired ellipse in the first quadrant:
# x(t) = M + A*(1 + cos(omega*t)), y(t) = M + B*(1 + sin(omega*t)).
A = 0.08  # x semi-axis [m]
B = 0.04  # y semi-axis [m]
WORKSPACE_MARGIN = 0.005  # [m]
PERIOD = 4.0  # [s]
SAMPLES = 1201


def inputs(time: float, model: CoreXYModel) -> NDArray[np.float64]:
    """Analytical input functions [u_x(t), u_y(t)]."""
    p = model.p
    omega = 2.0 * np.pi / PERIOD
    c = (2.0 * p.motor_inertia + p.mass * p.pulley_radius**2) / (
        p.torque_constant * p.pulley_radius
    )

    u_x = (
        c * A * p.inductance * omega**3 * np.sin(omega * time)
        - c * A * p.resistance * omega**2 * np.cos(omega * time)
        - 2.0 * p.back_emf_constant * A * omega / p.pulley_radius
        * np.sin(omega * time)
    )
    u_y = (
        -c * B * p.inductance * omega**3 * np.cos(omega * time)
        - c * B * p.resistance * omega**2 * np.sin(omega * time)
        + 2.0 * p.back_emf_constant * B * omega / p.pulley_radius
        * np.cos(omega * time)
    )
    return np.array([u_x, u_y])


def initial_state(model: CoreXYModel) -> NDArray[np.float64]:
    """State [x, x_dot, i_x, y, y_dot, i_y] at t=0."""
    p = model.p
    omega = 2.0 * np.pi / PERIOD
    c = (2.0 * p.motor_inertia + p.mass * p.pulley_radius**2) / (
        p.torque_constant * p.pulley_radius
    )
    return np.array([
        WORKSPACE_MARGIN + 2.0 * A,
        0.0,
        -c * A * omega**2,
        WORKSPACE_MARGIN + B,
        B * omega,
        0.0,
    ])


def main() -> None:
    configure_plots()
    model = CoreXYModel()
    time = np.linspace(0.0, PERIOD, SAMPLES)
    solution = solve_ivp(
        lambda t, state: model.derivative(t, state, inputs(t, model)),
        (time[0], time[-1]),
        initial_state(model),
        t_eval=time,
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    position = solution.y[[0, 3]]
    reference = np.vstack(
        (
            WORKSPACE_MARGIN + A * (1.0 + np.cos(2.0 * np.pi * time / PERIOD)),
            WORKSPACE_MARGIN + B * (1.0 + np.sin(2.0 * np.pi * time / PERIOD)),
        )
    )
    axis_inputs = np.column_stack([inputs(t, model) for t in time])

    output_directory = Path("outputs")
    output_directory.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=SQUARE_FIGURE_SIZE, constrained_layout=True)
    ax.plot(*reference, "--", label=r"$\mathrm{Analytical\ ellipse}$")
    ax.plot(*position, label=r"$\mathrm{Simulated\ model}$")
    ax.set(xlabel=r"$x\ [\mathrm{m}]$", ylabel=r"$y\ [\mathrm{m}]$")
    format_axes(ax, equal=True)
    ax.set(
        xlim=(0.0, 2.0 * A + 2.0 * WORKSPACE_MARGIN),
        ylim=(0.0, 2.0 * B + 2.0 * WORKSPACE_MARGIN),
    )
    ax.legend()
    path_destination = output_directory / "q1_4_ellipse_path.png"
    save_figure(fig, path_destination)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    ax.plot(time, position[0], label=r"$x$")
    ax.plot(time, position[1], label=r"$y$")
    ax.set(xlabel=r"$t\ [\mathrm{s}]$", ylabel=r"$\mathrm{Position}\ [\mathrm{m}]$")
    format_axes(ax)
    ax.legend()
    position_destination = output_directory / "q1_4_ellipse_position.png"
    save_figure(fig, position_destination)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    ax.plot(time, axis_inputs[0], label=r"$u_x$")
    ax.plot(time, axis_inputs[1], label=r"$u_y$")
    ax.set(xlabel=r"$t\ [\mathrm{s}]$", ylabel=r"$\mathrm{Voltage}\ [\mathrm{V}]$")
    format_axes(ax)
    ax.legend()
    inputs_destination = output_directory / "q1_4_ellipse_inputs.png"
    save_figure(fig, inputs_destination)
    plt.close(fig)

    maximum_error = np.max(np.linalg.norm(position - reference, axis=0))
    print(f"Saved {path_destination}")
    print(f"Saved {position_destination}")
    print(f"Saved {inputs_destination}")
    print(f"Maximum position error: {maximum_error:.3e} m")


if __name__ == "__main__":
    main()
