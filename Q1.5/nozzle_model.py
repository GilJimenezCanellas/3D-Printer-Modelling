from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp


Vector = NDArray[np.float64]
VoltageInput = float | Callable[[float], float]


@dataclass(frozen=True)
class NozzleParameters:

    resistance: float = 12.0  # R_n [ohm]
    inductance: float = 1.0e-3  # L_n [H]
    capacitance: float = 100.0e-6  # C_n [F]
    thermal_capacitance: float = 0.5  # C_t,n [J/K]
    thermal_resistance: float = 4.0  # R_t,n [K/W]
    ambient_temperature: float = 25.0  # T_e [degC]

    def __post_init__(self) -> None:
        positive = {
            "resistance": self.resistance,
            "inductance": self.inductance,
            "capacitance": self.capacitance,
            "thermal_capacitance": self.thermal_capacitance,
            "thermal_resistance": self.thermal_resistance,
        }


@dataclass(frozen=True)
class NozzleSolution:

    time: Vector
    state: NDArray[np.float64]

    @property
    def temperature(self) -> Vector:
        return self.state[:, 0]

    @property
    def capacitor_voltage(self) -> Vector:
        return self.state[:, 1]

    @property
    def inductor_current(self) -> Vector:
        return self.state[:, 2]


class NozzleModel:

    state_size = 3
    input_size = 1

    def __init__(self, parameters: NozzleParameters | None = None) -> None:
        self.p = parameters or NozzleParameters()

    def derivative(self, _time: float, state: ArrayLike, voltage: float) -> Vector:
        state_array = np.asarray(state, dtype=float)

        temperature, capacitor_voltage, inductor_current = state_array
        p = self.p

        temperature_dot = (
            capacitor_voltage**2 / p.resistance
            - (temperature - p.ambient_temperature) / p.thermal_resistance
        ) / p.thermal_capacitance
        capacitor_voltage_dot = (
            inductor_current - capacitor_voltage / p.resistance
        ) / p.capacitance
        inductor_current_dot = (float(voltage) - capacitor_voltage) / p.inductance

        return np.array(
            [temperature_dot, capacitor_voltage_dot, inductor_current_dot],
            dtype=float,
        )

    def simulate(
        self,
        voltage: VoltageInput,
        duration: float,
        initial_state: ArrayLike | None = None,
        output_step: float = 0.1,
    ) -> NozzleSolution:
        
        if initial_state is None:
            state_0 = np.array([self.p.ambient_temperature, 0.0, 0.0])
        else:
            state_0 = np.asarray(initial_state, dtype=float)

        voltage_function = voltage if callable(voltage) else lambda _time: float(voltage)
        output_time = np.arange(0.0, duration, output_step)
        output_time = np.append(output_time, duration)

        result = solve_ivp(
            lambda time, state: self.derivative(
                time, state, voltage_function(time)
            ),
            (0.0, duration),
            state_0,
            t_eval=output_time,
            method="Radau",
            rtol=1.0e-7,
            atol=1.0e-9,
        )
        if not result.success:
            raise RuntimeError(f"Nozzle simulation failed: {result.message}")

        return NozzleSolution(time=result.t, state=result.y.T)
