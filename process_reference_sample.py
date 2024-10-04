from tkinter.filedialog import askopenfilename
import tomllib
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import toml

from process_stack import process_stack
from multi_slice_viewer import (
    MultiSliceViewer,
    make_colored_overlay,
    RGBColorIndex,
)
from statistics_plots import particle_distributions
from files_io import (
    load_image_stack,
    check_config,
    Result,
    save_records,
)


def process_reference_sample(
    datapath: str | Path,
    view_stack: bool = True,
    view_distributions: bool = True,
) -> tuple:

    # * Checking and loading configuration
    datapath, dirpath, configpath = check_config(datapath)

    with open(configpath, "rb") as f:
        config = tomllib.load(f)

    # * Loading image stack
    image_stack, metadata = load_image_stack(
        str(datapath),
        zstart=config["zstart"],
        zstop=config["zstop"],
    )

    # * processing image stack
    results = process_stack(
        image_stack=image_stack,
        metadata=metadata,
        particle_diameter_um=config["particle_size_microns"],
        full_bboxes=False,
    )

    # * Displaying (optionnal) stack in MultiSliceViewer
    if view_stack:
        viewer = MultiSliceViewer(lognorm=True)
        viewer.plot(
            # volume=results["binary"],
            volume=image_stack,
            overlay=make_colored_overlay(
                volume=results["bboxes3d"],
                alpha=0.5,
                color_index=RGBColorIndex.RED,
            ),
        )
        plt.show()

    # * Computing particle concentration in number
    total_volume = np.float64(
        abs(metadata["z_coordinates"][-1] - metadata["z_coordinates"][0])
        * metadata["width"]
        * metadata["height"]
        * metadata["pixel_microns"] ** 2
    )

    num_particles = np.float64(len(results["props"]))

    particle_concentration = np.float64(num_particles / total_volume)


    # * Plotting particle size distribution
    histograms, plots = particle_distributions(
        results,
        metadata,
        bins_xy=20,
        bins_z=20,
        bins_pixels=100,
        bins_area=100,
        bins_diameter=100,
        boxplot_config={"showfliers": False},
        # verbose=True,
    )
    for key, plot in plots.items():
        plot.fig.savefig(
            fname=dirpath / f"{datapath.stem}_{key}_hist.png",
            format="png",
            # transparent=True,
        )

    # * Computing particle concentration in volume
    # ! le nombre de pixel médian est trop élevé pour obtenir un calcul
    # ! correct de la concentration en particules en volume pour les
    # ! traceurs de 0.2 microns.
    # ? Utiliser le fractile d'ordre 0.1 ou 0.2 ?
    median_num_pixels = np.median(
        np.array(
            [prop.num_pixels for prop in results["props"]],
        )
    )
    nums_pixels = np.array([prop.num_pixels for prop in results["props"]])
    num_particles_in_volume = np.sum(nums_pixels) / median_num_pixels
    particle_concentration_in_volume = num_particles_in_volume / total_volume

    if view_distributions:
        plt.show()

    # todo: add units and validation values from plots (medians)
    ref_results = {
        "zstep": Result(
            value=np.median(np.diff(np.array(metadata["z_coordinates"]))),
            unit="um",
        ),
        "total_volume": Result(total_volume, "um^3"),
        "num_particles_counted": Result(num_particles, "particles"),
        "num_particle_concentration": Result(
            value=particle_concentration,
            unit="particles/um^3",
        ),
        "vol_particles_counted": Result(
            value=num_particles_in_volume,
            unit="particles",
        ),
        "vol_particle_concentration": Result(
            value=particle_concentration_in_volume,
            unit="particles/um^3",
        ),
        "median_num_pixels": Result(median_num_pixels, "pixels"),  # ?
    }

    return image_stack, metadata, results, ref_results, histograms, plots


def print_reference_results(datapath:Path, results: dict):
    print(f"File: {datapath.name}")
    print(
        "Total volume analysed: "
        f"{results["total_volume"].value:.4e} "
        f"{results["total_volume"].unit}\n"
    )
    print("{:-^72}".format("Particle number calculations in number"))
    print(
        "Number of particles detected: "
        f"{results["num_particles_counted"].value:.4e} "
        f"{results["num_particles_counted"].unit}"
    )
    print(
        "Particle concentration: "
        f"{results["num_particle_concentration"].value:.4e} "
        f"{results["num_particle_concentration"].unit}\n"
    )
    
    print("{:-^72}".format("Particle number calculations in volume"))
    print(
        "Number of particles: "
        f"{results["vol_particles_counted"].value:.4e} "
        f"{results["vol_particles_counted"].unit}"
    )
    print(
        "Particle concentration: "
        f"{results["vol_particle_concentration"].value:.4e} "
        f"{results["vol_particle_concentration"].unit}\n"
    )


if __name__ == "__main__":
    datapath = Path(askopenfilename(title="Choose a data file"))
    # fmt: off
    image_stack, metadata, results, ref_results, histograms, plots =\
        process_reference_sample(datapath)
    # fmt: on
    print_reference_results(datapath, ref_results)
    save_results_path = save_records(
        [ref_results],
        datapath=datapath,
        suffix="reference_results",
    )
    print(f'Reference results saved at "{save_results_path}".')
