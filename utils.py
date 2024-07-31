from pathlib import Path

import numpy as np
import toml


def depth_of_field(
    *,
    numerical_aperture: float,
    magnification: float,
    smallest_dist: float,
    refraction_index: float = 1,  # for air
    wavelength: float = 550e-9,  # 550 nm in meters
) -> float:
    """Computes the approximate depth of field for a given set of
    arguments depending on the objective used. Formula is presented at
    https://www.microscopyu.com/tutorials/depthoffield

    Args:
        numerical_aperture (float): numerical aperture of the objective
        magnification (float): magification of the objective,
        eg. 10 for a 10x objective.
        smallest_dist (float): smallest distance that can be resolved
        by the detector that is placed in the image plane of the
        microscope objective.
        wavelength (float, optionnal): wavelength of the incident ligth in
        refraction_index (float, optionnal): refraction index of the media between
        the cover slip and the objective.
        1 for air, 1.515 for imersion oil, 1.333 for water.

    Returns:
        float: approximate depth of field in meters.
    """
    d_tot = (
        wavelength * refraction_index / numerical_aperture**2
        + refraction_index * smallest_dist / magnification / numerical_aperture
    )
    return d_tot


def z_resolution(
    diameter: float,
    num_pixels: float,
    x_resolution: float,
    y_resolution: float,
):
    """Computes the resolution along the z-axis for a that is to be used
    to make a shape spherical when it has been streched along the
    z-axis. This function is meant as a correction factor calculator.

    Args:
        diameter (float): The expected diameter of the sphere.
        num_pixels (float): The number of pixels included in the image
        of the sphere.
        x_resolution (float): The resolution along the x-axis.
        y_resolution (float): The resolution along the y-axis.

    Returns:
        float: The computed resolution along the z-axis.
    """
    # fmt: off
    return (
        (4 * np.pi * diameter ** 3)
        / (27 * num_pixels * x_resolution * y_resolution)
        )
    # fmt: on


def initialize_toml(filename: Path | str, metadata: dict):
    toml_dict = {
        "stack_start": metadata["z_levels"].start,
        "stack_stop": metadata["z_levels"].stop,
    }
    with open(filename, mode="w") as f:
        toml.dump(toml_dict, f)
