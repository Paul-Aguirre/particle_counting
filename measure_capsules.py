import tomllib
from pathlib import Path
from tkinter.filedialog import askopenfilename

import numpy as np

from files_io import Result, check_config, load_image_stack, save_results
from process_stack import process_stack


def measure_capsules(datapath: str | Path) -> list[dict]:
    # * getting back the selections from the config file
    datapath, _, configpath = check_config(datapath)
    with open(configpath, "rb") as f:
        config = tomllib.load(f)

    # * loading the substack from the main large image
    def get_selection_stack(config: dict, path: str | Path):
        for selection in config["selections"]:
            image_stack, metadata = load_image_stack(
                path=path,
                xstart=selection["xstart"],
                xstop=selection["xstop"],
                ystart=selection["ystart"],
                ystop=selection["ystop"],
                zstart=config["zstart"],
                zstop=config["zstop"],
            )
            yield image_stack, metadata

    capsules_results = []
    for substack, metadata in get_selection_stack(config=config, path=datapath):
        # * applying process stack to each substack
        results = process_stack(
            image_stack=substack,
            metadata=metadata,
            morph_open=False,
            full_bboxes=False,
            capsule=True,
            use_centroids=True,
        )
        # * extract capsule size parameters
        assert len(results["hull_props"]) == 1
        capsule_volume = results["hull_props"][0].area  # in µm^3
        capsule_diameter = results["hull_props"][0].equivalent_diameter_area
        # * and compute pectin concentration in capsule volume
        particles_in_hull = results["hull"] == results["binary"]
        particles_in_hull_results = process_stack(
            image_stack=particles_in_hull,
            metadata=metadata,
            binary=True,
            morph_open=False,
        )

        median_num_pixels = np.median(
            np.array(
                [prop.num_pixels for prop in particles_in_hull_results["scaled_props"]],
            )
        )
        nums_pixels = np.array([prop.num_pixels for prop in results["props"]])
        num_particles_in_volume = np.sum(nums_pixels) / median_num_pixels
        particle_concentration_in_volume = num_particles_in_volume / capsule_volume

        pectin_experimental_concentration = (
            particle_concentration_in_volume
            * config["pectin_concentration"]
            / config["particle_concentration"]
        )

        capsule_results = {
            "capsule_diameter": Result(capsule_diameter, "µm"),
            "capsule_volume": Result(capsule_volume, "µm^3"),
            "median_num_pixels": Result(median_num_pixels, "pixels"),
            "num_particles_in_volume": Result(num_particles_in_volume, "particles"),
            "particle_concentration_in_volume": Result(
                particle_concentration_in_volume, "particles/µm^3"
            ),
            "pectin_experimental_concentration": Result(
                pectin_experimental_concentration, "g/L"
            ),
        }
        capsules_results.append(capsule_results)

    return capsules_results


def print_capsule_results(results: list):
    for i, measurment in enumerate(results, start=1):
        title_str = f"Measurments for capsule n°{i}"
        print("{:-^72}".format(title_str))
        for key, value in measurment.items():
            line_title = key.replace("_", " ").capitalize()
            print(f"{line_title}: {value.value:.4e} {value.unit}")

    # * plot capsule size distribution
    # * plot pectin concentration in capsule against capsule radius


if __name__ == "__main__":
    datapath = Path(askopenfilename())
    capsules_results = measure_capsules(datapath)
    print_capsule_results(capsules_results)
    save_results(
        results=capsules_results,
        datapath=datapath,
        suffix="capsule_results",
    )
