from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FilamentParameters:
    density: float = 1240.0
    heat_capacity: float = 1800.0
    conductivity: float = 0.13
    width: float = 0.4e-3
    air_coefficient: float = 10.0
    bed_coefficient: float = 100.0
    ambient_temperature: float = 25.0
    bed_temperature: float = 60.0
    nozzle_temperature: float = 210.0

    @property
    def diffusivity(self) -> float:
        return self.conductivity / (self.density * self.heat_capacity)

    @property
    def cooling_rate(self) -> float:
        return (self.bed_coefficient + 3.0 * self.air_coefficient) / (
            self.density * self.heat_capacity * self.width
        )

    @property
    def equilibrium_temperature(self) -> float:
        return (
            self.bed_coefficient * self.bed_temperature
            + 3.0 * self.air_coefficient * self.ambient_temperature
        ) / (self.bed_coefficient + 3.0 * self.air_coefficient)


@dataclass(frozen=True)
class FilamentSolution:
    time: NDArray[np.float64]
    arc_length: NDArray[np.float64]
    temperature: NDArray[np.float64]
    print_time: float
    uniform_time: float


class FilamentCoolingModel:
    def __init__(
        self,
        length: float,
        print_speed: float,
        points: int = 300,
        parameters: FilamentParameters = FilamentParameters(),
    ) -> None:
        self.length = length
        self.print_speed = print_speed
        self.points = points
        self.p = parameters

    def simulate(
        self,
        uniformity_tolerance: float = 1.0,
        maximum_time: float = 180.0,
        output_step: float = 0.1,
    ) -> FilamentSolution:
        ds = self.length / self.points
        print_time = self.length / self.print_speed
        dt = min(0.02, 0.4 * ds**2 / self.p.diffusivity)
        temperature = np.full(self.points, np.nan)
        temperature[0] = self.p.nozzle_temperature
        times = [0.0]
        fields = [temperature.copy()]
        time = 0.0
        next_output = output_step

        while time < maximum_time:
            step = min(dt, maximum_time - time)
            new_time = time + step
            active = min(self.points, int(new_time / print_time * self.points) + 1)
            old_active = np.count_nonzero(~np.isnan(temperature))
            temperature[old_active:active] = self.p.nozzle_temperature

            if new_time < print_time:
                if active > 2:
                    current = temperature[:active].copy()
                    laplacian = np.empty(active)
                    laplacian[0] = 2.0 * (current[1] - current[0]) / ds**2
                    laplacian[1:-1] = (current[2:] - 2.0 * current[1:-1] + current[:-2]) / ds**2
                    laplacian[-1] = 0.0
                    current += step * self.p.diffusivity * laplacian
                    current += step * self.p.cooling_rate * (
                        self.p.equilibrium_temperature - current
                    )
                    current[-1] = self.p.nozzle_temperature
                    temperature[:active] = current
            else:
                temperature[np.isnan(temperature)] = self.p.nozzle_temperature
                laplacian = (
                    np.roll(temperature, -1)
                    - 2.0 * temperature
                    + np.roll(temperature, 1)
                ) / ds**2
                temperature += step * (
                    self.p.diffusivity * laplacian
                    + self.p.cooling_rate * (self.p.equilibrium_temperature - temperature)
                )

            time = new_time
            if time >= next_output or time >= print_time and times[-1] < print_time:
                times.append(time)
                fields.append(temperature.copy())
                next_output = time + output_step

            if time >= print_time and np.ptp(temperature) <= uniformity_tolerance:
                if times[-1] != time:
                    times.append(time)
                    fields.append(temperature.copy())
                break

        return FilamentSolution(
            np.asarray(times),
            np.arange(self.points) * ds,
            np.asarray(fields),
            print_time,
            time,
        )
