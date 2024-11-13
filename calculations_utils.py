import numpy as np


def compute_stack_total_volume(metadata: dict) -> np.float64:

    total_volume = np.float64(
        abs(metadata["z_coordinates"][-1] - metadata["z_coordinates"][0])
        * metadata["width"]
        * metadata["height"]
        * metadata["pixel_microns"] ** 2
    )
    return total_volume


def compute_particle_concentration_in_number(
    processing_results: dict,
    total_volume: float,
) -> tuple[np.float64, np.float64]:

    num_regions = np.float64(len(processing_results["props"]))
    particle_concentration_in_number = np.float64(num_regions / total_volume)
    return num_regions, particle_concentration_in_number


def compute_particle_concentration_in_volume(
    processing_results: dict,
    total_volume: float,
) -> tuple[np.float64, np.float64, np.float64]:

    # fmt: off
    nums_pixels = np.array([
        prop.num_pixels for prop in processing_results["props"]
        ])
    # fmt: on
    median_num_pixels = np.median(nums_pixels)
    num_particles_in_volume = np.sum(nums_pixels) / median_num_pixels
    particle_concentration_in_volume = num_particles_in_volume / total_volume

    return (
        median_num_pixels,
        num_particles_in_volume,
        particle_concentration_in_volume,
    )


def compute_pectin_concentration(
    particle_concentration: float,
    config: dict,
) -> float:
    pectin_experimental_concentration = (
        particle_concentration
        * config["pectin_concentration"]
        / config["particle_concentration"]
    )
    return pectin_experimental_concentration
