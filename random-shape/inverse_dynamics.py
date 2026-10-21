"""Inverse-dynamics utilities for prescribed Cartesian trajectories."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from corexy.model import CoreXYModel


Vector = NDArray[np.float64]


def current_per_acceleration(model: CoreXYModel) -> float:
    """Return C where i_x = C*x_ddot and i_y = C*y_ddot."""
    p = model.p
    return (2.0 * p.motor_inertia + p.mass * p.pulley_radius**2) / (
        p.torque_constant * p.pulley_radius
    )


def inverse_dynamics(
    model: CoreXYModel,
    velocity: ArrayLike,
    acceleration: ArrayLike,
    jerk: ArrayLike,
) -> Vector:
    """Calculate the decoupled voltage inputs [u_x, u_y]."""
    velocity = np.asarray(velocity, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    jerk = np.asarray(jerk, dtype=float)

    p = model.p
    c = current_per_acceleration(model)
    return (
        p.inductance * c * jerk
        + p.resistance * c * acceleration
        + 2.0 * p.back_emf_constant * velocity / p.pulley_radius
    )


def trajectory_state(
    model: CoreXYModel,
    position: ArrayLike,
    velocity: ArrayLike,
    acceleration: ArrayLike,
) -> Vector:
    """Build [x, x_dot, i_x, y, y_dot, i_y] on a trajectory."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)

    current = current_per_acceleration(model) * acceleration
    return np.array(
        [
            position[0],
            velocity[0],
            current[0],
            position[1],
            velocity[1],
            current[1],
        ]
    )
