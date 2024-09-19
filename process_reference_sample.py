from tkinter.filedialog import askopenfilename
import tomllib
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import toml

from process_stack import process_stack
from multi_slice_viewer import MultiSliceViewer, make_colored_overlay, RGBColorIndex
from particle_distributions import particle_distributions
from files_io import load_image_stack, check_config


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
        zstart=config["stack_start"],
        zstop=config["stack_stop"],
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
        viewer = MultiSliceViewer()
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
    total_volume = (
        abs(metadata["z_coordinates"][-1] - metadata["z_coordinates"][0])
        * metadata["width"]
        * metadata["height"]
        * metadata["pixel_microns"] ** 2
    )

    num_particles = len(results["props"])

    particle_concentration = num_particles / total_volume

    print("\n")
    print(f"File: {datapath.name}")
    print(f"Total volume analysed: {total_volume:.4e} µm^3")
    print("\n")
    print("{:-^72}".format("Particle number calculations in number"))
    print(f"Number of particles detected: {num_particles:.4e}")
    print(f"Particle concentration: {particle_concentration:.4e} particles/µm^3")

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

    print("\n")
    print("{:-^72}".format("Particle number calculations in volume"))
    # print("Without z-axis correction:")
    print(
        f"Number of particles: {num_particles_in_volume:.4e}",
    )
    print(
        "Particle concentration: "
        f"{particle_concentration_in_volume:.4e} particles/µm^3"
    )

    if view_distributions:
        plt.show()

    # * Dumping reference sample results in a file.
    # todo: add units and validation values from plots (medians)
    ref_results = {
        "zstep": np.median(np.diff(np.array(metadata["z_coordinates"]))),
        "total_volume": total_volume,
        "num_particles_counted": num_particles,
        "num_particle_concentration": particle_concentration,
        "vol_particles_counted": num_particles_in_volume,
        "vol_particle_concentration": particle_concentration_in_volume,
        "median_num_pixels": median_num_pixels,  # ?
    }
    ref_results_file = dirpath / f"{datapath.stem}_reference_results.toml"

    with open(ref_results_file, "w") as f:
        toml.dump(ref_results, f)

    return image_stack, metadata, results, histograms, plots


if __name__ == "__main__":
    datapath = Path(askopenfilename(title="Choose a data file"))
    # fmt: off
    image_stack, metadata, results, histograms, plots =\
        process_reference_sample(datapath)
    # fmt: on
