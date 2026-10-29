"""Seven-mode hybrid model of the printer during filament replacement."""

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
import sys

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "Q1.5"), str(ROOT / "random-shape")]

from corexy.model import CoreXYModel
from fourier_curve import FourierCurve
from inverse_dynamics import inverse_dynamics, trajectory_state
from nozzle_model import NozzleModel


Vector = NDArray[np.float64]


class Mode(IntEnum):
    PRINT = 0
    RETURN_HOME = 1
    WIPE = 2
    COOL = 3
    REPLACE = 4
    HEAT = 5
    MOVE_TO_START = 6


MODE_NAMES = {
    Mode.PRINT: "Print",
    Mode.RETURN_HOME: "Return home",
    Mode.WIPE: "Purge and wipe",
    Mode.COOL: "Cool nozzle",
    Mode.REPLACE: "Replace filament",
    Mode.HEAT: "Heat nozzle",
    Mode.MOVE_TO_START: "Move to start",
}


@dataclass(frozen=True)
class HybridParameters:
    printing_temperature: float = 210.0
    heating_voltage: float = 24.0
    cold_error: float = 1.0
    travel_time: float = 2.0
    position_gain: float = 200.0
    velocity_gain: float = 10.0
    tau_1: float = 12.0
    tau_2: float = 4.78
    tau_3: float = 1.0
    output_step: float = 0.05


@dataclass(frozen=True)
class Scenario:
    events: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class Transition:
    time: float
    source: Mode
    target: Mode | None
    guard: str
    reset: str


@dataclass(frozen=True)
class HybridSolution:
    time: Vector
    state: NDArray[np.float64]
    mode: NDArray[np.int64]
    reference: NDArray[np.float64]
    transitions: tuple[Transition, ...]

    @property
    def temperature(self) -> Vector:
        return self.state[:, 0]

    @property
    def x(self) -> Vector:
        return self.state[:, 3]

    @property
    def y(self) -> Vector:
        return self.state[:, 6]

    @property
    def tau(self) -> Vector:
        return self.state[:, 9]

    @property
    def saved_x(self) -> Vector:
        return self.state[:, 10]

    @property
    def saved_y(self) -> Vector:
        return self.state[:, 11]


def load_fourier_curve(path: Path) -> FourierCurve:
    table = np.loadtxt(path, delimiter=",", skiprows=1)
    return FourierCurve(
        table[:, 0].astype(int), table[:, 1] + 1j * table[:, 2]
    )


class HybridPrinter:
    """Hybrid automaton joining the nozzle and CoreXY state-space models.

    Continuous state:
    [T_n, u_C,n, i_L,n, x, x_dot, i_x, y, y_dot, i_y, tau, x_s, y_s].
    """

    def __init__(
        self,
        curve: FourierCurve,
        print_time: float = 8.0,
        parameters: HybridParameters = HybridParameters(),
    ) -> None:
        self.curve = curve
        self.print_time = print_time
        self.p = parameters
        self.motion = CoreXYModel()
        self.nozzle = NozzleModel()

    @property
    def holding_voltage(self) -> float:
        p_n = self.nozzle.p
        heat_loss = (
            self.p.printing_temperature - p_n.ambient_temperature
        ) / p_n.thermal_resistance
        return np.sqrt(heat_loss * p_n.resistance)

    def invariant(self, mode: Mode, state: Vector, local_time: float) -> bool:
        temperature = state[0]
        hot = np.isclose(temperature, self.p.printing_temperature, atol=1e-6)
        home = np.hypot(state[3], state[6]) <= 1e-6
        if mode == Mode.PRINT:
            return hot and local_time <= self.print_time
        if mode == Mode.RETURN_HOME:
            return hot and local_time <= self.p.travel_time
        if mode == Mode.WIPE:
            return hot and home and state[9] <= self.p.tau_3
        if mode == Mode.COOL:
            return temperature >= self.nozzle.p.ambient_temperature + self.p.cold_error
        if mode == Mode.REPLACE:
            return (
                temperature <= self.nozzle.p.ambient_temperature + self.p.cold_error
                and state[9] <= self.p.tau_2
            )
        if mode == Mode.HEAT:
            return temperature <= self.p.printing_temperature
        return hot and local_time <= self.p.travel_time

    def system(
        self,
        local_time: float,
        state: Vector,
        mode: Mode,
        motion_voltage: Vector | None = None,
        nozzle_voltage: float | None = None,
        tau_rate: float = 0.0,
    ) -> Vector:
        derivative = np.zeros(12)
        if nozzle_voltage is None:
            nozzle_voltage = 0.0 if mode == Mode.COOL else self.holding_voltage
        derivative[:3] = self.nozzle.derivative(
            local_time, state[:3], nozzle_voltage
        )
        if motion_voltage is not None:
            derivative[3:9] = self.motion.derivative(
                local_time, state[3:9], motion_voltage
            )
        derivative[9] = tau_rate
        return derivative

    def simulate(self, scenario: Scenario = Scenario()) -> HybridSolution:
        ambient = self.nozzle.p.ambient_temperature
        drawing_start = self.curve.time_derivatives(0.0, self.print_time)[0][0]
        state = np.array([
            ambient, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, drawing_start[0], drawing_start[1],
        ])
        time_parts: list[Vector] = []
        state_parts: list[NDArray[np.float64]] = []
        mode_parts: list[NDArray[np.int64]] = []
        reference_parts: list[NDArray[np.float64]] = []
        transitions: list[Transition] = []
        global_time = 0.0
        event_index = 0
        print_progress = 0.0

        def append(
            mode: Mode,
            local_time: Vector,
            states: NDArray[np.float64],
            reference: NDArray[np.float64] | None = None,
        ) -> None:
            nonlocal global_time, state
            start = 0 if not time_parts else 1
            time_parts.append(global_time + local_time[start:])
            state_parts.append(states[start:])
            mode_parts.append(np.full(len(local_time) - start, int(mode), dtype=int))
            if reference is None:
                reference = np.full((len(local_time), 2), np.nan)
            reference_parts.append(reference[start:])
            global_time += local_time[-1]
            state = states[-1].copy()

        def fixed_segment(
            mode: Mode,
            duration: float,
            dynamics,
            reference=None,
        ) -> None:
            count = max(2, int(np.ceil(duration / self.p.output_step)) + 1)
            local_time = np.linspace(0.0, duration, count)
            result = solve_ivp(
                dynamics,
                (0.0, duration),
                state,
                t_eval=local_time,
                method="Radau",
                rtol=1e-8,
                atol=1e-10,
            )
            append(mode, local_time, result.y.T, reference(local_time) if reference else None)

        def thermal_segment(mode: Mode, heating: bool) -> None:
            target = (
                self.p.printing_temperature
                if heating
                else ambient + self.p.cold_error
            )

            def event(_time, current):
                return current[0] - target

            event.terminal = True
            event.direction = 1 if heating else -1
            voltage = self.p.heating_voltage if heating else 0.0
            result = solve_ivp(
                lambda t, x: self.system(t, x, mode, nozzle_voltage=voltage),
                (0.0, 2000.0),
                state,
                events=event,
                dense_output=True,
                method="Radau",
                rtol=1e-8,
                atol=1e-10,
            )
            duration = result.t_events[0][0]
            count = max(2, int(np.ceil(duration / self.p.output_step)) + 1)
            local_time = np.linspace(0.0, duration, count)
            append(mode, local_time, result.sol(local_time).T)

        while True:
            thermal_segment(Mode.HEAT, True)
            transitions.append(Transition(
                global_time, Mode.HEAT, Mode.MOVE_TO_START,
                "T_n >= T_set", "T_n := T_set; heater := hold",
            ))
            state[:3] = self._thermal_hold_state()

            start_position = state[10:12].copy()
            fixed_segment(
                Mode.MOVE_TO_START,
                self.p.travel_time,
                lambda t, x: self.system(
                    t, x, Mode.MOVE_TO_START,
                    self._pd_voltage(x, start_position),
                ),
                lambda times: np.repeat(
                    start_position[None, :], len(times), axis=0
                ),
            )
            transitions.append(Transition(
                global_time, Mode.MOVE_TO_START, Mode.PRINT,
                "t_6 >= travel_time", "x_m := (x_s,y_s)",
            ))

            if event_index < len(scenario.events):
                _, event_fraction = scenario.events[event_index]
                print_end = event_fraction * self.print_time
                print_speed = 1.0
            else:
                print_end = self.print_time
                print_speed = 2.0
            print_duration = (print_end - print_progress) / print_speed
            position, velocity, acceleration, _ = self.curve.time_derivatives(
                print_progress, self.print_time
            )
            state[3:9] = trajectory_state(
                self.motion,
                position[0],
                print_speed * velocity[0],
                print_speed**2 * acceleration[0],
            )

            def print_data(t):
                position_t, velocity_t, acceleration_t, jerk_t = (
                    self.curve.time_derivatives(
                        print_progress + print_speed * t, self.print_time
                    )
                )
                return (
                    position_t,
                    print_speed * velocity_t,
                    print_speed**2 * acceleration_t,
                    print_speed**3 * jerk_t,
                )

            def print_dynamics(t, x):
                _, velocity_t, acceleration_t, jerk_t = print_data(t)
                voltage = inverse_dynamics(
                    self.motion, velocity_t[0], acceleration_t[0], jerk_t[0]
                )
                return self.system(t, x, Mode.PRINT, voltage, tau_rate=1.0)

            fixed_segment(
                Mode.PRINT,
                print_duration,
                print_dynamics,
                lambda times: print_data(times)[0],
            )
            print_progress = print_end
            if event_index == len(scenario.events):
                transitions.append(Transition(
                    global_time, Mode.PRINT, None,
                    "t_0 >= print_time", "finished := true",
                ))
                break

            event_kind, _ = scenario.events[event_index]
            reason = (
                "change_colour = true"
                if event_kind == "colour"
                else "tau >= tau_1"
            )
            transitions.append(Transition(
                global_time, Mode.PRINT, Mode.RETURN_HOME, reason,
                "(x_s,y_s) := (x,y); printing := false; tau := 0",
            ))
            state[10:12] = [state[3], state[6]]
            state[9] = 0.0
            event_index += 1
            home = np.zeros(2)
            fixed_segment(
                Mode.RETURN_HOME,
                self.p.travel_time,
                lambda t, x: self.system(
                    t, x, Mode.RETURN_HOME, self._pd_voltage(x, home)
                ),
                lambda times: np.repeat(home[None, :], len(times), axis=0),
            )
            state[3:9] = 0.0
            transitions.append(Transition(
                global_time, Mode.RETURN_HOME, Mode.WIPE,
                "(x,y) = (0,0)", "motion state := 0",
            ))

            fixed_segment(
                Mode.WIPE,
                self.p.tau_3,
                lambda t, x: self.system(
                    t, x, Mode.WIPE, tau_rate=1.0
                ),
            )
            transitions.append(Transition(
                global_time, Mode.WIPE, Mode.COOL,
                "tau >= tau_3", "u_n := 0; tau := 0",
            ))
            state[9] = 0.0

            thermal_segment(Mode.COOL, False)
            state[1:3] = 0.0
            transitions.append(Transition(
                global_time, Mode.COOL, Mode.REPLACE,
                "T_n <= T_e + Delta_T_cold", "electrical state := 0",
            ))
            fixed_segment(
                Mode.REPLACE,
                self.p.tau_2,
                lambda t, x: self.system(
                    t, x, Mode.REPLACE, nozzle_voltage=0.0, tau_rate=1.0
                ),
            )
            state[9] = 0.0
            transitions.append(Transition(
                global_time, Mode.REPLACE, Mode.HEAT,
                "tau >= tau_2", "tau := 0; new filament loaded",
            ))

        return HybridSolution(
            np.concatenate(time_parts),
            np.vstack(state_parts),
            np.concatenate(mode_parts),
            np.vstack(reference_parts),
            tuple(transitions),
        )

    def _thermal_hold_state(self) -> Vector:
        voltage = self.holding_voltage
        return np.array([
            self.p.printing_temperature,
            voltage,
            voltage / self.nozzle.p.resistance,
        ])


    def _pd_voltage(self, state: Vector, target: Vector) -> Vector:
        """Return Cartesian PD voltages from measured position and velocity."""
        position = state[[3, 6]]
        velocity = state[[4, 7]]
        return (
            self.p.position_gain * (target - position)
            - self.p.velocity_gain * velocity
        )
