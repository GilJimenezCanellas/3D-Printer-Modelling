"""Silhouette extraction, arc-length sampling, and planar Fourier curves."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from skimage import filters, measure, morphology


RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def load_silhouette(path: Path) -> tuple[RealArray, NDArray[np.bool_]]:
    """Load an image and identify the largest foreground component.

    The less prevalent side of Otsu's threshold is assumed to be the silhouette,
    so both black-on-white and white-on-black inputs are accepted.
    """
    gray = np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0
    threshold = filters.threshold_otsu(gray)
    bright = gray > threshold
    foreground = bright if np.mean(bright) < 0.5 else ~bright
    foreground = morphology.remove_small_objects(foreground, max_size=15)
    labels = measure.label(foreground)
    regions = measure.regionprops(labels)
    if not regions:
        raise ValueError(f"No silhouette found in {path}")
    largest = max(regions, key=lambda region: region.area)
    return gray, labels == largest.label


def ordered_contour(mask: NDArray[np.bool_]) -> RealArray:
    """Return the longest ordered boundary as Cartesian pixel coordinates."""
    padded = np.pad(mask, 1, constant_values=False)
    contours = measure.find_contours(padded.astype(float), 0.5)
    if not contours:
        raise ValueError("The silhouette has no extractable contour")
    row_column = max(contours, key=len) - 1.0
    contour = np.column_stack((row_column[:, 1], -row_column[:, 0]))
    if np.linalg.norm(contour[0] - contour[-1]) > 1e-12:
        contour = np.vstack((contour, contour[0]))
    return contour


def resample_by_arclength(contour: RealArray, samples: int) -> tuple[RealArray, RealArray]:
    """Resample a closed contour uniformly in normalized cumulative arc length."""
    if samples < 8:
        raise ValueError("At least eight contour samples are required")
    segment_length = np.linalg.norm(np.diff(contour, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_length)))
    if cumulative[-1] == 0.0:
        raise ValueError("The contour has zero length")
    parameter = cumulative / cumulative[-1]
    uniform_parameter = np.arange(samples, dtype=float) / samples
    resampled = np.column_stack(
        [np.interp(uniform_parameter, parameter, contour[:, axis]) for axis in range(2)]
    )
    return uniform_parameter, resampled


def scale_to_workspace(points: RealArray, size: float) -> RealArray:
    """Place a scaled curve above and right of the lower-left origin."""
    lower_left = points - points.min(axis=0)
    span = np.ptp(lower_left, axis=0).max()
    if span == 0.0:
        raise ValueError("Cannot scale a zero-size contour")
    margin = 5.0e-3
    return margin + lower_left * ((size - 2.0 * margin) / span)


@dataclass(frozen=True)
class FourierCurve:
    """Complex Fourier series z(s) = x(s) + j*y(s), with s in [0, 1)."""

    frequencies: NDArray[np.int64]
    coefficients: ComplexArray

    @classmethod
    def fit(cls, points: RealArray, order: int) -> "FourierCurve":
        signal = points[:, 0] + 1j * points[:, 1]
        spectrum = np.fft.fft(signal) / len(signal)
        frequencies = np.arange(-order, order + 1, dtype=int)
        coefficients = spectrum[frequencies % len(signal)]
        return cls(frequencies, coefficients)

    def evaluate(self, parameter: ArrayLike, derivative: int = 0) -> RealArray:
        """Evaluate the curve or a derivative with respect to normalized arc time."""
        parameter = np.atleast_1d(np.asarray(parameter, dtype=float))
        angular = 2j * np.pi * self.frequencies
        basis = np.exp(2j * np.pi * np.outer(parameter, self.frequencies))
        values = basis @ (self.coefficients * angular**derivative)
        return np.column_stack((values.real, values.imag))

    def time_derivatives(self, time: ArrayLike, period: float) -> tuple[RealArray, ...]:
        """Return position, velocity, acceleration, and jerk at time values."""
        time = np.atleast_1d(np.asarray(time, dtype=float))
        parameter = np.mod(time / period, 1.0)
        return tuple(
            self.evaluate(parameter, derivative) / period**derivative
            for derivative in range(4)
        )

    def save(self, path: Path) -> None:
        table = np.column_stack(
            (self.frequencies, self.coefficients.real, self.coefficients.imag)
        )
        np.savetxt(
            path, table, delimiter=",", header="frequency,real,imag", comments=""
        )
