import numpy as np

from files_io import Result


def compute_stack_total_volume(metadata: dict) -> np.float64:

    total_volume = np.float64(
        abs(metadata["z_coordinates"][-1] - metadata["z_coordinates"][0])
        * metadata["width"]
        * metadata["height"]
        * metadata["pixel_microns"] ** 2
    )
    return total_volume  # in um^3


def compute_particle_concentration_in_number(
    total_volume: float,
    num_regions: float,
) -> tuple[np.float64, np.float64]:

    # num_regions = np.float64(len(processing_results["props"]))
    particle_concentration_in_number = np.float64(num_regions / total_volume)
    return num_regions, particle_concentration_in_number


def compute_particle_concentration_in_volume(
    nums_pixels: np.ndarray,
    total_volume: float,
) -> tuple[np.float64, np.float64, np.float64]:

    median_num_pixels = np.median(nums_pixels)
    num_particles_in_volume = np.sum(nums_pixels) / median_num_pixels
    particle_concentration_in_volume = num_particles_in_volume / total_volume

    return (
        median_num_pixels,
        num_particles_in_volume,
        particle_concentration_in_volume,
    )


def compute_pectin_concentration(
    particle_current_concentration: float,
    particle_init_concentration: float,
    pectin_init_concentration: float,
) -> float:
    # fmt: off
    pectin_exp_concentration = (
        particle_current_concentration 
        * pectin_init_concentration 
        / particle_init_concentration
    )
    # fmt: on
    return pectin_exp_concentration


def compute_capsule_concentrations(
    num_regions: float,
    nums_pixels: np.ndarray,
    capsule_diameter: float,
    capsule_volume: float,
    particle_th_ref_concentration: float,
    particle_exp_ref_concentration: float,
    pectin_th_concentration: float,
    calcium_chloride_concentration: float,
) -> dict:

    # * compute particle concentration in number
    num_particles_in_number, particle_concentration_in_number = (
        compute_particle_concentration_in_number(
            total_volume=capsule_volume,
            num_regions=num_regions,
        )
    )

    # * compute particle concentration in capsule volume
    (
        median_num_pixels,
        num_particles_in_volume,
        particle_concentration_in_volume,
    ) = compute_particle_concentration_in_volume(
        total_volume=capsule_volume,
        nums_pixels=nums_pixels,
    )

    # * compute pectin concentrations in number/volume using
    # * th/exp particle reference concentration
    pectin_exp_concentration_th_part_in_number = compute_pectin_concentration(
        particle_current_concentration=particle_concentration_in_number,
        particle_init_concentration=particle_th_ref_concentration,
        pectin_init_concentration=pectin_th_concentration,
    )
    pectin_exp_concentration_th_part_in_volume = compute_pectin_concentration(
        particle_current_concentration=particle_concentration_in_volume,
        particle_init_concentration=particle_th_ref_concentration,
        pectin_init_concentration=pectin_th_concentration,
    )
    pectin_exp_concentration_exp_part_in_number = compute_pectin_concentration(
        particle_current_concentration=particle_concentration_in_number,
        particle_init_concentration=particle_exp_ref_concentration,
        pectin_init_concentration=pectin_th_concentration,
    )
    pectin_exp_concentration_exp_part_in_volume = compute_pectin_concentration(
        particle_current_concentration=particle_concentration_in_volume,
        particle_init_concentration=particle_exp_ref_concentration,
        pectin_init_concentration=pectin_th_concentration,
    )

    # * normalising pectin concentration
    # fmt: off
    pectin_exp_concentration_th_part_num_norm = (
        pectin_exp_concentration_th_part_in_number
        / pectin_th_concentration
    )
    pectin_exp_concentration_th_part_vol_norm = (
        pectin_exp_concentration_th_part_in_volume
        / pectin_th_concentration
    )
    pectin_exp_concentration_exp_part_num_norm = (
        pectin_exp_concentration_exp_part_in_number
        / pectin_th_concentration
    )
    pectin_exp_concentration_exp_part_vol_norm = (
        pectin_exp_concentration_exp_part_in_volume
        / pectin_th_concentration
    )
    # fmt: on

    # * Collecting and organising the results
    capsule_record = {
        "capsule_diameter": Result(capsule_diameter, "um"),
        "capsule_volume": Result(capsule_volume, "um^3"),
        "median_num_pixels": Result(median_num_pixels, "pixels"),
        "num_particles_in_number": Result(
            num_particles_in_number,
            "particles",
        ),
        "particle_concentration_in_number": Result(
            particle_concentration_in_number,
            "particles/um^3",
        ),
        "num_particles_in_volume": Result(
            num_particles_in_volume,
            "particles",
        ),
        "particle_concentration_in_volume": Result(
            particle_concentration_in_volume, "particles/um^3"
        ),
        "pectin_experimental_concentration_th_part_in_number": Result(
            pectin_exp_concentration_th_part_in_number, "g/L"
        ),
        "pectin_experimental_concentration_th_part_in_volume": Result(
            pectin_exp_concentration_th_part_in_volume, "g/L"
        ),
        "pectin_experimental_concentration_exp_part_in_number": Result(
            pectin_exp_concentration_exp_part_in_number, "g/L"
        ),
        "pectin_experimental_concentration_exp_part_in_volume": Result(
            pectin_exp_concentration_exp_part_in_volume, "g/L"
        ),
        "pectin_th_concentration": Result(
            np.float64(pectin_th_concentration),
            "g/L",
        ),
        "pectin_exp_concs_th_part_num_norm": Result(
            pectin_exp_concentration_th_part_num_norm,
            "",
        ),
        "pectin_exp_concs_th_part_vol_norm": Result(
            pectin_exp_concentration_th_part_vol_norm,
            "",
        ),
        "pectin_exp_concs_exp_part_num_norm": Result(
            pectin_exp_concentration_exp_part_num_norm,
            "",
        ),
        "pectin_exp_concs_exp_part_vol_norm": Result(
            pectin_exp_concentration_exp_part_vol_norm,
            "",
        ),
        "calcium_chloride_concentration": Result(
            np.float64(calcium_chloride_concentration), "mmol/L"
        ),
    }

    return capsule_record


def recompute_pectin_concentrations(
    old_record: dict,
    particle_th_ref_concentration: float,
    particle_exp_ref_concentration: float,
    pectin_th_concentration: float,
    calcium_chloride_concentration: float,
) -> dict:

    new_record = old_record.copy()

    pectin_experimental_concentration_th_part_in_number = compute_pectin_concentration(
        particle_current_concentration=old_record[
            "particle_concentration_in_number"
        ].value,
        particle_init_concentration=particle_th_ref_concentration,
        pectin_init_concentration=pectin_th_concentration,
    )
    pectin_experimental_concentration_th_part_in_volume = compute_pectin_concentration(
        particle_current_concentration=old_record[
            "particle_concentration_in_volume"
        ].value,
        particle_init_concentration=particle_th_ref_concentration,
        pectin_init_concentration=pectin_th_concentration,
    )
    pectin_experimental_concentration_exp_part_in_number = compute_pectin_concentration(
        particle_current_concentration=old_record[
            "particle_concentration_in_number"
        ].value,
        particle_init_concentration=particle_exp_ref_concentration,
        pectin_init_concentration=pectin_th_concentration,
    )
    pectin_experimental_concentration_exp_part_in_volume = compute_pectin_concentration(
        particle_current_concentration=old_record[
            "particle_concentration_in_volume"
        ].value,
        particle_init_concentration=particle_exp_ref_concentration,
        pectin_init_concentration=pectin_th_concentration,
    )

    # fmt: off
    pectin_exp_concs_th_part_num_norm = (
        pectin_experimental_concentration_th_part_in_number
        / pectin_th_concentration
    )
    pectin_exp_concs_th_part_vol_norm = (
        pectin_experimental_concentration_th_part_in_volume
        / pectin_th_concentration
    )
    pectin_exp_concs_exp_part_num_norm = (
        pectin_experimental_concentration_exp_part_in_number
        / pectin_th_concentration
    )
    pectin_exp_concs_exp_part_vol_norm = (
        pectin_experimental_concentration_exp_part_in_volume
        / pectin_th_concentration
    )
    # fmt: on

    new_values = {
        "pectin_experimental_concentration_th_part_in_number": Result(
            pectin_experimental_concentration_th_part_in_number, "g/L"
        ),
        "pectin_experimental_concentration_th_part_in_volume": Result(
            pectin_experimental_concentration_th_part_in_volume, "g/L"
        ),
        "pectin_experimental_concentration_in_number": Result(
            pectin_experimental_concentration_exp_part_in_number, "g/L"
        ),
        "pectin_experimental_concentration_in_volume": Result(
            pectin_experimental_concentration_exp_part_in_volume, "g/L"
        ),
        "pectin_th_concentration": Result(
            np.float64(pectin_th_concentration),
            "g/L",
        ),
        "pectin_exp_concs_th_part_num_norm": Result(
            pectin_exp_concs_th_part_num_norm,
            "",
        ),
        "pectin_exp_concs_th_part_vol_norm": Result(
            pectin_exp_concs_th_part_vol_norm,
            "",
        ),
        "pectin_exp_concs_exp_part_num_norm": Result(
            pectin_exp_concs_exp_part_num_norm,
            "",
        ),
        "pectin_exp_concs_exp_part_vol_norm": Result(
            pectin_exp_concs_exp_part_vol_norm,
            "",
        ),
        "calcium_chloride_concentration": Result(
            np.float64(calcium_chloride_concentration), "mmol/L"
        ),
    }
    new_record.update(new_values)

    return new_record
