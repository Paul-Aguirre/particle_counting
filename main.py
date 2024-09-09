from enum import Enum
from tkinter.filedialog import askopenfilename
import tomllib
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from process_stack import process_stack
from multi_slice_viewer import MultiSliceViewer, make_colored_overlay, RGBColorIndex
from particle_distributions import particle_distributions
from files_inputs import load_image_stack, check_config


# class RGBColorIndex(Enum):
#     RED = (0,)
#     GREEN = (1,)
#     BLUE = (2,)
#     YELLOW = 0, 1
#     PURPLE = 0, 2
#     CYAN = 1, 2
#     WHITE = 0, 1, 2


# def make_bbox_overlay(
#     bboxes: np.ndarray,
#     alpha: float,
#     color_index: int | RGBColorIndex,
# ):
#     bbox_overlay = np.zeros(
#         (*bboxes.shape, 4),
#         dtype=np.float64,
#     )
#     for i in color_index:
#         bbox_overlay[..., i] = bboxes
#     bbox_overlay[..., 3] = alpha

#     return bbox_overlay


def main(
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
        start=config["stack_start"],
        stop=config["stack_stop"],
    )

    # * processing image stack
    results = process_stack(
        image_stack=image_stack,
        metadata=metadata,
        particle_diameter=config["particle_size_microns"],
        full_bbox=False,
    )

    # * Displaying (optionnal) stack in MultiSliceViewer
    if view_stack:
        viewer = MultiSliceViewer()
        viewer.plot(
            # volume=results["binary"],
            volume=image_stack,
            overlay=make_colored_overlay(
                bboxes=results["bboxes3d"],
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

    return image_stack, metadata, results, histograms, plots


if __name__ == "__main__":
    datapath = Path(askopenfilename(title="Choose a data file"))
    image_stack, metadata, results, histograms, plots = main(datapath)
