from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt


def particle_distributions(
    results: dict,
    metadata: dict,
    bins_xy: int | Sequence | str = 10,
    bins_z: int | Sequence | str = 10,
    bins_pixels: int | Sequence | str = 10,
    bins_area: int | Sequence | str = 10,
    bins_diameter: int | Sequence | str = 10,
    boxplot_config: dict = {
        "showcaps": True,
        "showbox": True,
        "showfliers": True,
    },
):
    """Plots the distributions of characteristic values of the
    particles. Characteristic values are dimensions along axes x, y and
    z on one plot and number of pixels, the "area" (in scikit image
    definition's) and the area equivalent diameter. distributions are
    plotted as a histogram, as a cumulative distribution plot and
    a boxplot.

    Args:
        results (dict): the results dictionnary returned by process_stack()
        metadata (dict): the metadata dictionnary returned by
        load_image_stack().
        bins_xy (int | Sequence | str, optional): bins for x and y
        histograms. Defaults to 10.
        bins_z (int | Sequence | str, optional): bins for z histogram.
        Defaults to 10.
        bins_pixels (int | Sequence | str, optional): bins for
        num_pixels histogram. Defaults to 10.
        bins_area (int | Sequence | str, optional): bins for area
        histogram. Defaults to 10.
        bins_diameter (int | Sequence | str, optional): bins for
        diameter histogram. Defaults to 10.
    """
    x = []
    y = []
    z = []
    areas = []
    nums_pixels = []
    diameters = []

    for prop in results["props"]:
        z.append(prop.image.shape[0])
        y.append(prop.image.shape[1])
        x.append(prop.image.shape[2])
        areas.append(prop.area)
        nums_pixels.append(prop.num_pixels)
        diameters.append(prop.equivalent_diameter_area)

    x = np.array(x, dtype=np.float32) * metadata["pixel_microns"]
    y = np.array(y, dtype=np.float32) * metadata["pixel_microns"]
    z = np.array(z, dtype=np.float32) * np.abs(
        np.mean(np.diff(metadata["z_coordinates"]))
    )
    areas = np.array(areas, dtype=np.float32)
    nums_pixels = np.array(nums_pixels, dtype=np.float32)
    diameters = np.array(diameters, dtype=np.float32)

    fig1, axs1 = plt.subplots(3, 3)
    # fig1.tight_layout()
    axs1[0, 0].hist(nums_pixels, bins=bins_pixels)
    axs1[0, 0].set_title("number of pixels")
    axs1[0, 0].set_ylabel("count")
    axs1[1, 0].plot(
        np.sort(nums_pixels),
        np.arange(1, len(nums_pixels) + 1) / len(nums_pixels),
        # cumulative frequencies
    )
    axs1[2, 0].boxplot(nums_pixels, **boxplot_config)

    axs1[0, 1].hist(areas, bins=bins_area)
    axs1[0, 1].set_title("area")
    axs1[1, 1].plot(
        np.sort(areas),
        np.arange(1, len(areas) + 1) / len(areas),
    )
    axs1[2, 1].boxplot(areas, **boxplot_config)

    axs1[0, 2].hist(diameters, bins=bins_diameter)
    axs1[0, 2].set_title("equivalent area diameter")
    axs1[1, 2].plot(
        np.sort(diameters),
        np.arange(1, len(diameters) + 1) / len(diameters),
    )
    axs1[2, 2].boxplot(diameters, **boxplot_config)

    fig2, axs2 = plt.subplots(3, 3)
    gs = axs2[2, 0].get_gridspec()
    for ax in axs2[2, :2]:
        ax.remove()
    axs2boxplot_xy = fig2.add_subplot(gs[2, :2])
    # fig2.tight_layout()

    # TODO: scalling x, y & z by the resolution for better lisibility
    # TODO: print the tables of the cumulated frequency graphs to get exact values
    axs2[0, 0].hist(x, bins=bins_xy)
    axs2[0, 0].set_title("x")
    axs2[0, 0].set_ylabel("count")
    axs2[1, 0].plot(
        np.sort(x),
        np.arange(1, len(x) + 1) / len(x),
        marker="o",
        linestyle="-",
    )
    axs2[1, 0].set_ylabel("cumulated frequencies")

    axs2[0, 1].hist(y, bins=bins_xy)
    axs2[0, 1].set_title("y")
    axs2[0, 1].set_xlabel("Size in microns")
    axs2[1, 1].plot(
        np.sort(y),
        np.arange(1, len(y) + 1) / len(y),
        marker="o",
        linestyle="-",
    )

    axs2[0, 2].hist(z, bins=bins_z)
    axs2[0, 2].set_title("z")
    axs2[1, 2].plot(
        np.sort(z),
        np.arange(1, len(z) + 1) / len(z),
        marker="o",
        linestyle="-",
    )
    axs2[2, 2].boxplot(z, tick_labels=["z"], **boxplot_config)

    axs2boxplot_xy.boxplot([x, y], tick_labels=["x", "y"], **boxplot_config)
    axs2boxplot_xy.set_ylabel("Size in microns")

    plt.show()
