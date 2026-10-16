"""Q1.4: propagate the decoupled model using analytical ellipse inputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from corexy.model import CoreXYModel


# Desired ellipse: x(t) = A*cos(omega*t), y(t) = B*sin(omega*t).
A = 0.08  # x semi-axis [m]
B = 0.04  # y semi-axis [m]
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
    return np.array([A, 0.0, -c * A * omega**2, 0.0, B * omega, 0.0])


def main() -> None:
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
            A * np.cos(2.0 * np.pi * time / PERIOD),
            B * np.sin(2.0 * np.pi * time / PERIOD),
        )
    )
    axis_inputs = np.column_stack([inputs(t, model) for t in time])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].plot(*reference, "--", label="analytical ellipse")
    axes[0].plot(*position, label="simulated model")
    axes[0].set(
        xlabel="x [m]", ylabel="y [m]", title="Tool-head path", aspect="equal"
    )
    axes[0].legend()

    axes[1].plot(time, position[0], label="x")
    axes[1].plot(time, position[1], label="y")
    axes[1].set(xlabel="time [s]", ylabel="position [m]", title="Position")
    axes[1].legend()

    axes[2].plot(time, axis_inputs[0], label=r"$u_x$")
    axes[2].plot(time, axis_inputs[1], label=r"$u_y$")
    axes[2].set(xlabel="time [s]", ylabel="voltage [V]", title="Axis inputs")
    axes[2].legend()

    output_directory = Path("outputs")
    output_directory.mkdir(exist_ok=True)
    destination = output_directory / "q1_4_ellipse.png"
    fig.savefig(destination, dpi=180)
    plt.show()

    maximum_error = np.max(np.linalg.norm(position - reference, axis=0))
    print(f"Saved {destination}")
    print(f"Maximum position error: {maximum_error:.3e} m")


if __name__ == "__main__":
    main()
