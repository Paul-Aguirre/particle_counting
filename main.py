from tkinter.filedialog import askopenfilename
import tomllib
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from process_stack import process_stack
from multi_slice_viewer import MultiSliceViewer
from particle_distributions import particle_distributions
from files_inputs import load_image_stack, check_config


def main(
    datapath: str | Path,
    view_stack: bool = True,
    view_distributions: bool = True,
) -> None:

    datapath, dirpath, configpath = check_config(datapath)

    with open(configpath, "rb") as f:
        config = tomllib.load(f)

    image_stack, metadata = load_image_stack(
        str(datapath),
        start=config["stack_start"],
        stop=config["stack_stop"],
    )

    results = process_stack(
        image_stack=image_stack,
        metadata=metadata,
        particle_diameter=config["particle_size_microns"],
        full_bbox=False,
    )

    if view_stack:
        viewer = MultiSliceViewer()
        viewer.plot(
            # volume=results["binary"],
            volume=image_stack,
            bboxes=results["bboxes3d"],
            bbox_alpha=0.5,
        )
        plt.show()

    total_volume = (
        abs(metadata["z_coordinates"][-1] - metadata["z_coordinates"][0])
        * metadata["width"]
        * metadata["height"]
        * metadata["pixel_microns"] ** 2
    )

    num_particles = len(results["props"])

    particle_concentration = num_particles / total_volume

    print(f"File: {datapath.name}")
    print(f"Total volume analysed: {total_volume:.4e} µm^3")
    print("{:-^72}".format("Particle number calculations in number"))
    print(f"Number of particles detected: {num_particles:.4e}")
    print(f"Particle concentration: {particle_concentration:.4e} particles/µm^3")

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

    # particle number calculations in volume
    # ! le nombre de pixel médian est trop élevé pour obtenir un calcul
    # ! correct de la concentration en particules en volume pour les
    # ! traceurs de 0.2 microns.
    # ? Utiliser le fractile d'ordre 0.1 ou 0.2
    median_num_pixels = np.median(
        np.array(
            [prop.num_pixels for prop in results["props"]],
        )
    )
    nums_pixels = np.array([prop.num_pixels for prop in results["props"]])
    num_particles_in_volume = np.sum(nums_pixels) / median_num_pixels
    particle_concentration_in_volume = num_particles_in_volume / total_volume

    print("{:-^72}".format("Particle number calculations in volume"))
    # print("Without z-axis correction:")
    print(
        f"Number of particles (computed in volume): {num_particles_in_volume:.4e}",
    )
    print(
        "Particle concentration (computed in volume):"
        f"{particle_concentration_in_volume:.4e} particles/µm^3"
    )

    if view_distributions:
        plt.show()

    return image_stack, metadata, results, histograms, plots


if __name__ == "__main__":
    datapath = Path(askopenfilename(title="Choose a data file"))
    image_stack, metadata, results, histograms, plots = main(datapath)
