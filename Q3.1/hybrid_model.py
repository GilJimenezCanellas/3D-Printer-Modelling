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
from inverse_dynamics import inverse_dynamics
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


class ParkingState(IntEnum):
    INITIAL = 0
    LEFT_DETECTED = 1
    BOTTOM_DETECTED = 2
    PARKED = 3
    ERROR = 4


class ParkingCommand(IntEnum):
    VERTICAL = 0
    HORIZONTAL = 1


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
    temperature_tolerance: float = 0.1
    position_tolerance: float = 1.0e-6
    velocity_tolerance: float = 1.0e-5
    maximum_positioning_time: float = 30.0
    position_gain: float = 200.0
    tau_1: float = 12.0
    tau_3: float = 4.78
    tau_2: float = 1.0
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
    parking_state: NDArray[np.int64]
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


@dataclass(frozen=True)
class MotorMeasurements:
    """Measured motor angles and angular velocities."""

    angles: Vector
    angular_velocities: Vector


class ParkingDFA:
    """Detect arrival at both home boundaries from commanded motion and speed."""

    def __init__(self) -> None:
        self.state = ParkingState.PARKED

    def reset(self) -> None:
        self.state = ParkingState.INITIAL

    def update(
        self,
        command: ParkingCommand,
        angular_velocities: Vector,
        angular_velocity_tolerance: float,
    ) -> ParkingState:
        moving = bool(
            np.any(np.abs(angular_velocities) > angular_velocity_tolerance)
        )
        if self.state == ParkingState.INITIAL:
            if not moving:
                self.state = (
                    ParkingState.BOTTOM_DETECTED
                    if command == ParkingCommand.VERTICAL
                    else ParkingState.LEFT_DETECTED
                )
        elif self.state == ParkingState.BOTTOM_DETECTED:
            if command != ParkingCommand.HORIZONTAL:
                self.state = ParkingState.ERROR
            elif not moving:
                self.state = ParkingState.PARKED
        elif self.state == ParkingState.LEFT_DETECTED:
            if command != ParkingCommand.VERTICAL:
                self.state = ParkingState.ERROR
            elif not moving:
                self.state = ParkingState.PARKED
        return self.state


    @property
    def parked(self) -> bool:
        return self.state == ParkingState.PARKED


@dataclass(frozen=True)
class ControlInputs:
    """Exogenous voltage inputs applied to the common plant dynamics."""

    nozzle_voltage: float
    motion_voltage: Vector


class PrinterController:
    """Output-feedback controller driven by measurements and the FSM mode."""

    def __init__(
        self,
        motion: CoreXYModel,
        nozzle: NozzleModel,
        parameters: HybridParameters,
    ) -> None:
        self.motion = motion
        self.nozzle = nozzle
        self.p = parameters
        self.saved_position = np.zeros(2)

    @property
    def holding_voltage(self) -> float:
        heat_loss = (
            self.p.printing_temperature - self.nozzle.p.ambient_temperature
        ) / self.nozzle.p.thermal_resistance
        return np.sqrt(heat_loss * self.nozzle.p.resistance)

    def position(self, measurements: MotorMeasurements) -> Vector:
        """Reconstruct Cartesian position from measured motor angles."""
        left, right = measurements.angles
        radius = self.motion.p.pulley_radius
        return 0.5 * radius * np.array([left + right, left - right])

    def velocity(self, measurements: MotorMeasurements) -> Vector:
        """Reconstruct Cartesian velocity from measured motor speeds."""
        left, right = measurements.angular_velocities
        radius = self.motion.p.pulley_radius
        return 0.5 * radius * np.array([left + right, left - right])

    def latch_position(self, measurements: MotorMeasurements) -> None:
        """Store the reconstructed position before leaving S0."""
        self.saved_position = self.position(measurements)

    def inputs(
        self,
        mode: Mode,
        measurements: MotorMeasurements,
        print_kinematics: tuple[Vector, Vector, Vector] | None = None,
        motion_reference: Vector | None = None,
    ) -> ControlInputs:
        """Calculate nozzle and Cartesian motor voltages."""
        nozzle_voltage = (
            0.0
            if mode in (Mode.COOL, Mode.REPLACE)
            else self.holding_voltage
        )
        if mode in (Mode.RETURN_HOME, Mode.MOVE_TO_START):
            if motion_reference is None:
                motion_reference = (
                    np.zeros(2)
                    if mode == Mode.RETURN_HOME
                    else self.saved_position
                )
            motion_voltage = self.p.position_gain * (
                motion_reference - self.position(measurements)
            )
        elif mode == Mode.PRINT:
            if print_kinematics is None:
                raise ValueError("print kinematics are required in S0")
            velocity, acceleration, jerk = print_kinematics
            motion_voltage = inverse_dynamics(
                self.motion, velocity, acceleration, jerk
            )
        else:
            motion_voltage = np.zeros(2)
        return ControlInputs(nozzle_voltage, motion_voltage)


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
        self.controller = PrinterController(
            self.motion, self.nozzle, self.p
        )
        self.parking = ParkingDFA()

    def measurements(self, state: Vector) -> MotorMeasurements:
        """Apply the hybrid output map to the Cartesian motion state."""
        radius = self.motion.p.pulley_radius
        x, x_dot, _, y, y_dot, _ = state[3:9]
        return MotorMeasurements(
            np.array([(x + y) / radius, (x - y) / radius]),
            np.array([(x_dot + y_dot) / radius, (x_dot - y_dot) / radius]),
        )

    def invariant(self, mode: Mode, state: Vector, local_time: float) -> bool:
        temperature = state[0]
        hot = temperature >= (
            self.p.printing_temperature - self.p.temperature_tolerance
        )
        home = np.hypot(state[3], state[6]) <= self.p.position_tolerance
        if mode == Mode.PRINT:
            return hot and state[9] <= self.p.tau_1
        if mode == Mode.RETURN_HOME:
            return hot and self.parking.state not in (
                ParkingState.PARKED, ParkingState.ERROR
            )
        if mode == Mode.WIPE:
            return hot and home and state[9] <= self.p.tau_2
        if mode == Mode.COOL:
            return temperature >= self.nozzle.p.ambient_temperature + self.p.temperature_tolerance
        if mode == Mode.REPLACE:
            return (
                temperature <= self.nozzle.p.ambient_temperature + self.p.temperature_tolerance
                and state[9] <= self.p.tau_3
            )
        if mode == Mode.HEAT:
            return temperature <= (
                self.p.printing_temperature - self.p.temperature_tolerance
            )
        measurements = self.measurements(state)
        position_error = np.max(np.abs(
            self.controller.position(measurements)
            - self.controller.saved_position
        ))
        speed = np.max(np.abs(self.controller.velocity(measurements)))
        return hot and (
            position_error > self.p.position_tolerance
            or speed > self.p.velocity_tolerance
        )

    def system(
        self,
        local_time: float,
        state: Vector,
        inputs: ControlInputs,
    ) -> Vector:
        """Common continuous dynamics used in every discrete mode."""
        derivative = np.zeros(12)
        derivative[:3] = self.nozzle.derivative(
            local_time, state[:3], inputs.nozzle_voltage
        )
        derivative[3:9] = self.motion.derivative(
            local_time, state[3:9], inputs.motion_voltage
        )
        derivative[9] = 1.0
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
        self.controller.saved_position = drawing_start.copy()
        time_parts: list[Vector] = []
        state_parts: list[NDArray[np.float64]] = []
        mode_parts: list[NDArray[np.int64]] = []
        reference_parts: list[NDArray[np.float64]] = []
        parking_parts: list[NDArray[np.int64]] = []
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
            parking_parts.append(
                np.full(len(local_time) - start, int(self.parking.state), dtype=int)
            )
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

        def arrival_event(target: Vector, axes: tuple[int, ...]):
            def event(_time, current):
                measurements = self.measurements(current)
                position_error = np.abs(
                    self.controller.position(measurements)[list(axes)]
                    - target[list(axes)]
                ) / self.p.position_tolerance
                speed = np.abs(
                    self.controller.velocity(measurements)[list(axes)]
                ) / self.p.velocity_tolerance
                return max(np.max(position_error), np.max(speed)) - 1.0

            event.terminal = True
            event.direction = -1
            return event

        def positioning_segment(
            mode: Mode,
            target: Vector,
            axes: tuple[int, ...],
        ) -> None:
            event = arrival_event(target, axes)
            result = solve_ivp(
                lambda t, x: self.system(
                    t,
                    x,
                    self.controller.inputs(
                        mode, self.measurements(x), motion_reference=target
                    ),
                ),
                (0.0, self.p.maximum_positioning_time),
                state,
                events=event,
                dense_output=True,
                method="Radau",
                rtol=1e-8,
                atol=1e-10,
            )
            if len(result.t_events[0]) == 0:
                raise RuntimeError(f"{mode.name} did not reach its target")
            duration = result.t_events[0][0]
            count = max(2, int(np.ceil(duration / self.p.output_step)) + 1)
            local_time = np.linspace(0.0, duration, count)
            reference = np.repeat(target[None, :], count, axis=0)
            append(mode, local_time, result.sol(local_time).T, reference)

        def thermal_segment(mode: Mode, heating: bool) -> None:
            target = (
                self.p.printing_temperature - self.p.temperature_tolerance
                if heating
                else ambient + self.p.temperature_tolerance
            )

            def event(_time, current):
                return current[0] - target

            event.terminal = True
            event.direction = 1 if heating else -1
            result = solve_ivp(
                lambda t, x: self.system(
                    t, x, self.controller.inputs(mode, self.measurements(x))
                ),
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
                "T_n >= T_set - epsilon_T",
                "controller: u_n := u_set; tau := 0",
            ))
            state[9] = 0.0
            self.parking.reset()
            start_position = self.controller.saved_position.copy()
            positioning_segment(
                Mode.MOVE_TO_START, start_position, (0, 1)
            )
            transitions.append(Transition(
                global_time, Mode.MOVE_TO_START, Mode.PRINT,
                "(x,y) = (x_s,y_s) and (x_dot,y_dot) = (0,0)",
                "tau := 0",
            ))
            state[9] = 0.0

            if event_index < len(scenario.events):
                _, event_fraction = scenario.events[event_index]
                print_end = event_fraction * self.print_time
                print_speed = 1.0
            else:
                print_end = self.print_time
                print_speed = 2.0
            print_duration = (print_end - print_progress) / print_speed
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
                inputs = self.controller.inputs(
                    Mode.PRINT,
                    self.measurements(x),
                    (velocity_t[0], acceleration_t[0], jerk_t[0]),
                )
                return self.system(t, x, inputs)

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
                "controller: (x_s,y_s) := (x,y); printing := false; tau := 0",
            ))
            self.controller.latch_position(self.measurements(state))
            state[9] = 0.0
            state[10:12] = self.controller.saved_position
            event_index += 1
            self.parking.reset()
            home = np.zeros(2)
            positioning_segment(Mode.RETURN_HOME, home, (0, 1))

            angular_tolerance = (
                2.0 * self.p.velocity_tolerance
                / self.motion.p.pulley_radius
            )
            stopped_measurements = self.measurements(state)
            self.parking.update(
                ParkingCommand.VERTICAL,
                stopped_measurements.angular_velocities,
                angular_tolerance,
            )
            self.parking.update(
                ParkingCommand.HORIZONTAL,
                stopped_measurements.angular_velocities,
                angular_tolerance,
            )
            parking_parts[-1][-1] = int(self.parking.state)
            if not self.parking.parked:
                raise RuntimeError("parking DFA did not reach q_P")

            transitions.append(Transition(
                global_time, Mode.RETURN_HOME, Mode.WIPE,
                "q = q_P and (x,y) = (0,0)", "tau := 0",
            ))
            state[9] = 0.0

            fixed_segment(
                Mode.WIPE,
                self.p.tau_2,
                lambda t, x: self.system(
                    t, x, self.controller.inputs(Mode.WIPE, self.measurements(x))
                ),
            )
            transitions.append(Transition(
                global_time, Mode.WIPE, Mode.COOL,
                "tau >= tau_2", "controller: u_n := 0; tau := 0",
            ))
            state[9] = 0.0

            thermal_segment(Mode.COOL, False)
            transitions.append(Transition(
                global_time, Mode.COOL, Mode.REPLACE,
                "T_n <= T_e + epsilon_T",
                "tau := 0",
            ))
            state[9] = 0.0
            fixed_segment(
                Mode.REPLACE,
                self.p.tau_3,
                lambda t, x: self.system(
                    t, x, self.controller.inputs(Mode.REPLACE, self.measurements(x))
                ),
            )
            transitions.append(Transition(
                global_time, Mode.REPLACE, Mode.HEAT,
                "tau >= tau_3", "new filament loaded; tau := 0",
            ))
            state[9] = 0.0

        return HybridSolution(
            np.concatenate(time_parts),
            np.vstack(state_parts),
            np.concatenate(mode_parts),
            np.vstack(reference_parts),
            np.concatenate(parking_parts),
            tuple(transitions),
        )
