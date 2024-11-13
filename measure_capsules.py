import tomllib
from pathlib import Path
from tkinter.filedialog import askopenfilename
import argparse
from typing import Literal

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import toml

# from memory_profiler import profile

from calculations_utils import (
    compute_particle_concentration_in_number,
    compute_particle_concentration_in_volume,
    compute_pectin_concentration,
)
from files_io import (
    Result,
    ResultRecord,
    check_config,
    df_to_csv,
    get_save_path,
    get_selection_stack,
    save_records,
)
from plane import Plane
from process_stack import process_stack
from statistics_plots import (
    capsule_distrib_kde,
    capsule_scatter,
    particle_distributions,
)


# @profile
def measure_capsules(
    datapath: str | Path,
    reader: Literal["nd2reader", "nd2"],
) -> tuple[list[ResultRecord], pd.DataFrame]:

    print(f"Analysing '{str(datapath)}'.")

    # * getting back the selections from the config file
    datapath, dirpath, configpath = check_config(datapath)
    with open(configpath, "rb") as f:
        config = tomllib.load(f)

    if not config["selections"]:
        raise ValueError("No selected region in the stack.")

    # * loading the substack from the main large image
    capsules_records: list = []
    # capsules_processing_results = []
    results: dict = {}
    n_iter: int = 1
    for substack, metadata in get_selection_stack(
        config=config,
        datapath=datapath,
        reader=reader,
        use_patches=True,
    ):

        # * applying process stack to each substack
        process_stack(
            image_stack=substack,
            metadata=metadata,
            results=results,
            threshold=config.get("threshold"),
            threshold_by_image=config.get("threshold_by_image", False),
            morph_open=False,
            bboxes=False,
            full_bboxes=False,
            capsule=True,
            use_centroids=True,
        )
        if not config.get("threshold"):
            config["threshold"] = int(results["threshold"])
            with open(configpath, "w") as f:
                toml.dump(config, f)
        # capsules_processing_results.append(results)

        # * Plotting processing steps:
        # - initial image
        fig_initial, ax_initial = plt.subplots()
        initial_middle = substack[Plane.XY.middle_slice(substack.shape[0])]
        ax_initial.imshow(
            initial_middle,
            cmap="grey",
            norm="log",
        )
        fig_initial.savefig(
            fname=dirpath / f"{datapath.stem}_initial_middle_no-{n_iter}.png",
            format="png",
        )

        # - binary
        binary_middle = results["binary"][
            Plane.XY.middle_slice(
                results["binary"].shape[0],
            )
        ]
        fig_binary, ax_binary = plt.subplots()
        ax_binary.imshow(
            binary_middle,
            cmap="grey",
        )
        fig_binary.savefig(
            fname=dirpath / f"{datapath.stem}_binary_middle_no-{n_iter}.png",
            format="png",
        )

        # - initial + convex hull
        hull_middle = results["hull"][
            Plane.XY.middle_slice(
                results["hull"].shape[0],
            )
        ]
        ax_initial.imshow(
            hull_middle,
            cmap="viridis",
            alpha=0.5,
        )
        fig_initial.savefig(
            fname=dirpath / f"{datapath.stem}_hull_middle_no-{n_iter}.png",
            format="png",
        )

        # * Plotting particle size distribution
        histograms, plots = particle_distributions(
            results,
            metadata,
            bins_xy="auto",
            bins_z="auto",
            bins_pixels="auto",
            bins_area="auto",
            bins_diameter="auto",
            boxplot_config={"showfliers": False},
            # verbose=True,
        )
        for key, plot in plots.items():
            plot.fig.savefig(
                fname=dirpath / f"{datapath.stem}_{key}_hist_no-{n_iter}.png",
                format="png",
                # transparent=True,
            )

        # * extract capsule size parameters
        assert len(results["hull_props"]) == 1
        capsule_volume = results["hull_props"][0].area  # in µm^3
        # (area for 3D object is volume, checked in scikit image source code)
        capsule_diameter = results["hull_props"][0].equivalent_diameter_area
        # (also checked in skimage source code that formula applies to 3D case)

        # * compute pectin concentration in number
        num_particles_in_number, particle_concentration_in_number = (
            compute_particle_concentration_in_number(
                processing_results=results,
                total_volume=capsule_volume,
            )
        )
        # num_particles_in_number = np.float64(len(results["props"]))
        # particle_concentration_in_number = num_particles_in_number / capsule_volume

        # * and compute pectin concentration in capsule volume
        # two methods have been thought of for accessing only the tracers
        # in the hull:
        # - the die cutting approach:
        # particles_in_hull = results["hull"] == results["binary"]

        # - the centroid position check approach:
        particles_in_hull = np.zeros(substack.shape)

        for particle in results["props"]:
            # testing if tracer centroid is in the hull
            centroid_coords = tuple(int(coord) for coord in particle.centroid)
            if results["hull"][centroid_coords]:
                # extracting bbox dimensions/position to fill tracer
                # image back in it
                mind, minr, minc, maxd, maxr, maxc = particle.bbox
                depths = np.arange(mind, maxd)
                rows = np.arange(minr, maxr)
                cols = np.arange(minc, maxc)
                dgrid, rgrid, cgrid = np.meshgrid(
                    depths,
                    rows,
                    cols,
                    indexing="ij",
                )
                particles_in_hull[dgrid, rgrid, cgrid] = particle.image

        # the actual computations:
        particles_in_hull_results = process_stack(
            image_stack=particles_in_hull,
            metadata=metadata,
            binary=True,
            morph_open=False,
        )
        del particles_in_hull

        (
            median_num_pixels,
            num_particles_in_volume,
            particle_concentration_in_volume,
        ) = compute_particle_concentration_in_volume(
            processing_results=particles_in_hull_results,
            total_volume=capsule_volume,
        )

        # fmt: off
        pectin_experimental_concentration_in_number = (
            compute_pectin_concentration(
                particle_concentration=particle_concentration_in_number,
                config=config,
            )
        )
        pectin_experimental_concentration_in_volume = (
            compute_pectin_concentration(
                particle_concentration=particle_concentration_in_volume,
                config=config,
            )
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
            "pectin_experimental_concentration_in_number": Result(
                pectin_experimental_concentration_in_number, "g/L"
            ),
            "pectin_experimental_concentration_in_volume": Result(
                pectin_experimental_concentration_in_volume, "g/L"
            ),
        }

        capsules_records.append(capsule_record)
        n_iter += 1

    # fmt: off
    capsules_records_values = [
        {k: v.value for k, v in record.items()}
        for record in capsules_records
    ]
    # fmt: on

    df_capsules_records = pd.DataFrame.from_records(capsules_records_values)

    df_capsules_records["c_0"] = config["pectin_concentration"]

    # fmt: off
    df_capsules_records["pectin_exp_concs_norm"] = (
        df_capsules_records.particle_concentration_in_volume
        / df_capsules_records.c_0
    )
    # fmt: on

    return (
        capsules_records,
        df_capsules_records,
    )  # capsules_processing_results


def print_capsule_records(records_lst: list[ResultRecord]):
    for i, measurment in enumerate(records_lst, start=1):
        print("")
        title_str = f"Measurments for capsule n°{i}"
        print("{:-^72}".format(title_str))
        for key, value in measurment.items():
            line_title = key.replace("_", " ").capitalize()
            print(f"{line_title}: {value.value:.4e} {value.unit}")

    # * plot capsule size distribution
    # * plot pectin concentration in capsule against capsule radius


def plot_save_capsules_stats(
    datapath: str | Path,
    df_capsules_records: pd.DataFrame,
) -> None:

    hist, kde = capsule_distrib_kde(df_records=df_capsules_records)

    # kde value computation for each datapoint has to be done
    # on a complete dataset, meaning that:
    # if a sample is split in several data files, the measurements
    # have to be regrouped in a single DataFrame before computing
    # kde values.
    df_capsules_records["caps_diams_kde"] = gaussian_kde(
        df_capsules_records.capsule_diameter
    ).evaluate(df_capsules_records.capsule_diameter)

    scatterplot = capsule_scatter(df_records=df_capsules_records)

    hist.savefig(fname=get_save_path(datapath, "hist_kde", "png"))
    kde.savefig(fname=get_save_path(datapath, "kde", "png"))
    scatterplot.savefig(fname=get_save_path(datapath, "scatter", "png"))

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(prog="measure_capsules")
    parser.add_argument(
        "-d",
        "--datapath",
        action="store",
        required=False,
        default=None,
        type=Path,
        help="The path to the datafile to analyse.",
    )
    parser.add_argument(
        "-r",
        "--reader",
        action="store",
        required=False,
        choices=["nd2", "nd2reader"],
        default="nd2reader",
        help="The reader to use for openning ND2 files. 'nd2'"
        " option is necessarry for 8-bit files.",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Shows the plots in pyplot windows.",
    )
    args = parser.parse_args()
    if args.datapath is None:
        args.datapath = Path(askopenfilename())

    datapath, *_ = check_config(args.datapath)
    (
        capsules_records,
        df_capsules_records,
        # capsules_processing_results,
    ) = measure_capsules(datapath=datapath, reader=args.reader)

    print_capsule_records(capsules_records)

    if args.show_plots:
        plt.show()

    save_records_path = save_records(
        records_lst=capsules_records,
        datapath=datapath,
        suffix="capsule_records",
    )
    save_records_path_csv = df_to_csv(
        df_records=df_capsules_records,
        datapath=datapath,
        suffix="capsule_records",
    )
    print(
        f'\nCapsule measurements results saved at \n"{save_records_path}"'
        f'\n and at "\n{save_records_path_csv}".'
    )


if __name__ == "__main__":
    main()
