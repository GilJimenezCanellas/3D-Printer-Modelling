"""Decoupled Cartesian state-space model of the CoreXY mechanism."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


Vector = NDArray[np.float64]


@dataclass(frozen=True)
class CoreXYParameters:
    """SI parameters of the print head and two identical DC motors."""

    mass: float = 0.50
    pulley_radius: float = 0.010
    motor_inertia: float = 1.0e-4
    resistance: float = 2.0
    inductance: float = 5.0e-3
    torque_constant: float = 0.10
    back_emf_constant: float = 0.10


class CoreXYModel:
    """Decoupled continuous-time model.

    State: [x, x_dot, i_x, y, y_dot, i_y]
    Input: [u_x, u_y]

    """

    state_size = 6
    input_size = 2

    def __init__(self, parameters: CoreXYParameters | None = None) -> None:
        self.p = parameters or CoreXYParameters()

    def derivative(self, _time: float, state: ArrayLike, inputs: ArrayLike) -> Vector:
        state = np.asarray(state, dtype=float)
        inputs = np.asarray(inputs, dtype=float)

        _, x_dot, i_x, _, y_dot, i_y = state
        u_x, u_y = inputs
        p = self.p

        acceleration_gain = (
            p.torque_constant * p.pulley_radius
            / (2.0 * p.motor_inertia + p.mass * p.pulley_radius**2)
        )
        x_ddot = acceleration_gain * i_x
        y_ddot = acceleration_gain * i_y
        i_x_dot = (
            u_x
            - p.resistance * i_x
            - 2.0 * p.back_emf_constant * x_dot / p.pulley_radius
        ) / p.inductance
        i_y_dot = (
            u_y
            - p.resistance * i_y
            - 2.0 * p.back_emf_constant * y_dot / p.pulley_radius
        ) / p.inductance

        return np.array([x_dot, x_ddot, i_x_dot, y_dot, y_ddot, i_y_dot])
