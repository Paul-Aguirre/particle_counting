from typing import Sequence, NamedTuple

import matplotlib.axes
import matplotlib.figure
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


class HistogramData(NamedTuple):
    data: np.ndarray
    cumdata: np.ndarray
    bins: np.ndarray


class DistributionPlot(NamedTuple):
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes


def hist_freq_cum(
    ax: matplotlib.axes.Axes,
    data: list | np.ndarray,
    bins: int | Sequence | str | np.ndarray | None = 10,
    title: str | None = None,
) -> tuple:
    """Plots histogramm and cumulative frequency plot on a given
    matplotlib axes. Adds a secondary y axis to read the frequency on.

    Args:
        ax (matplotlib.axes.Axes): The axes onto which the histogram is
        plotted.
        data (list | numpy.ndarray): Data used for tracing the histogram.
        bins (int | Sequence | str | numpy.ndarray, optional): Histogram bins. Passed
        down to ax.hist() method. Defaults to 10.
        title (str | None, optional): Sets the title of the ax.
        Defaults to None.

    Returns:
        tuple: A tuple containing:
            - the cumulated values of the histogram, not normalised.
            - the bins edges
            - the secondary y axis
    """
    _, bins, _ = ax.hist(
        data,
        bins=bins,
    )
    cum_data, *_ = ax.hist(
        data,
        bins=bins,
        cumulative=True,
        histtype="step",
        color="red",
    )
    secax = ax.secondary_yaxis(
        "right",
        functions=(
            lambda x: x / np.max(cum_data),
            lambda x: x * np.max(cum_data),
        ),
    )
    ax.grid(True)
    if title:
        ax.set_title(title)
    return cum_data, bins, secax


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
    verbose: bool = False,
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
        verbose (bool, optionnal): If True, prints the values of the
        cumulative frequency plot. Defaults to False.
    """

    # Saved for later use:
    # ------------------------------------------------------------------
    # results_key: str = "props",

    # for prop in results[results_key]:

    # results_key(str, optional): key at which the desired region
    # properties are stored in the results dictionnary.
    # Defaults to "props".
    # ------------------------------------------------------------------

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

    # normalizing x, y and z for readability
    x = np.array(x, dtype=np.float32) * metadata["pixel_microns"]
    y = np.array(y, dtype=np.float32) * metadata["pixel_microns"]
    z = np.array(z, dtype=np.float32) * np.abs(
        np.mean(np.diff(metadata["z_coordinates"]))
    )
    areas = np.array(areas, dtype=np.float32)
    nums_pixels = np.array(nums_pixels, dtype=np.float32)
    diameters = np.array(diameters, dtype=np.float32)

    # figsize = (15.2, 8.7)
    # figsize = (6.4, 4.8)
    figsize = (9.6, 7.2)
    fig1, axs1 = plt.subplots(2, 3, figsize=figsize)

    cum_num_pixels, bins_pixels, _ = hist_freq_cum(
        axs1[0, 0], nums_pixels, bins_pixels, "number of pixels"
    )
    axs1[0, 0].set_ylabel("count")
    axs1[1, 0].boxplot(
        nums_pixels,
        tick_labels=["number of pixels"],
        **boxplot_config,
    )

    cum_area, bins_area, _ = hist_freq_cum(axs1[0, 1], areas, bins_area, "area")
    axs1[1, 1].boxplot(areas, tick_labels=["area"], **boxplot_config)

    cum_diameter, bins_diameter, secax_diameter = hist_freq_cum(
        axs1[0, 2], diameters, bins_diameter, "equivalent area diameter"
    )
    secax_diameter.set_ylabel("Cumulative frequency")
    axs1[1, 2].boxplot(diameters, tick_labels=["diameter"], **boxplot_config)
    fig1.set_layout_engine(layout="tight")

    fig2, axs2 = plt.subplots(2, 3, figsize=figsize)

    cum_x, bins_xy, _ = hist_freq_cum(axs2[0, 0], x, bins_xy, "x")
    axs2[0, 0].set_ylabel("count")

    cum_y, bins_xy, _ = hist_freq_cum(axs2[0, 1], y, bins_xy, "y")
    axs2[0, 1].set_xlabel("Size in microns")

    cum_z, bins_z, secax_z = hist_freq_cum(axs2[0, 2], z, bins_z, "z")
    secax_z.set_ylabel("cumulated frequencies")
    axs2[1, 2].boxplot(z, tick_labels=["z"], **boxplot_config)

    # plotting x and y boxplots together because their values are close
    gs = axs2[1, 0].get_gridspec()
    for ax in axs2[1, :2]:
        ax.remove()
    axs2boxplot_xy = fig2.add_subplot(gs[1, :2])
    axs2boxplot_xy.boxplot([x, y], tick_labels=["x", "y"], **boxplot_config)
    axs2boxplot_xy.set_ylabel("Size in microns")
    fig2.set_layout_engine(layout="tight")

    histograms = {
        "num_pixels": HistogramData(nums_pixels, cum_num_pixels, bins_pixels),
        "area": HistogramData(areas, cum_area, bins_area),
        "diameter": HistogramData(diameters, cum_diameter, bins_diameter),
        "x": HistogramData(x, cum_x, bins_xy),
        "y": HistogramData(y, cum_y, bins_xy),
        "z": HistogramData(z, cum_z, bins_z),
    }

    plots = {
        "pix_area_diam": DistributionPlot(fig1, axs1),
        "xyz": DistributionPlot(fig2, axs2),
    }

    # plt.show()
    if verbose:
        binss = [
            bins_pixels,
            bins_area,
            bins_diameter,
            bins_xy,
            bins_xy,
            bins_z,
        ]
        cum_hists = [
            cum_num_pixels,
            cum_area,
            cum_diameter,
            cum_x,
            cum_y,
            cum_z,
        ]
        measurement_strs = [
            "num_pixels",
            "area",
            "diameter",
            "x",
            "y",
            "z",
        ]
        for bins, cum_hist, measurement_str in zip(
            binss,
            cum_hists,
            measurement_strs,
        ):
            print(measurement_str)

    return histograms, plots


def print_histogram(histogram: HistogramData) -> None:
    print("bins, cumulative frequency, frequency")
    print(
        np.concatenate(
            [
                histogram.bins[:-1, np.newaxis],
                histogram.cumdata[:, np.newaxis] / np.max(histogram.cumdata),
                histogram.data[:, np.newaxis] / np.max(histogram.data),
            ],
            axis=1,
        )
    )
