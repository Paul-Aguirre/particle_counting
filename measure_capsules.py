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
    compute_capsule_concentrations,
    compute_particle_concentration_in_number,
    compute_particle_concentration_in_volume,
    compute_pectin_concentration,
    recompute_pectin_concentrations,
)
from files_io import (
    Result,
    ResultRecord,
    check_config,
    df_to_csv,
    get_save_path,
    get_selection_stack,
    load_records,
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
    convert_to_8bit: bool = False,
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
        convert_to_8bit=convert_to_8bit,
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

        particles_in_hull_results = process_stack(
            image_stack=particles_in_hull,
            metadata=metadata,
            binary=True,
            morph_open=False,
        )
        del particles_in_hull

        # compute all concentrations values
        capsule_record = compute_capsule_concentrations(
            num_regions=np.float64(len(results["props"])),
            nums_pixels=np.array(
                [prop.num_pixels for prop in particles_in_hull_results["props"]]
            ),
            capsule_diameter=capsule_diameter,
            capsule_volume=capsule_volume,
            particle_th_concentration=config["particle_concentration"],
            pectin_th_concentration=config["pectin_concentration"],
            calcium_chloride_concentration=config["calcium_chloride_concentration"],
        )
        capsules_records.append(capsule_record)
        n_iter += 1

    # fmt: off
    capsules_records_values = [
        {k: v.value for k, v in record.items()}
        for record in capsules_records
    ]
    # fmt: on

    df_capsules_records = pd.DataFrame.from_records(capsules_records_values)

    return (
        capsules_records,
        df_capsules_records,
    )  # capsules_processing_results


def remake_records(path: Path | str) -> tuple[dict, pd.DataFrame]:

    # * getting back the selections from the config file
    datapath, dirpath, configpath = check_config(path)
    print(f"Analysing '{str(datapath)}'.")
    with open(configpath, "rb") as f:
        config = tomllib.load(f)

    old_records_path = get_save_path(
        datapath=datapath, suffix="capsule_records", ext="toml"
    )
    records_list = load_records(path=old_records_path)

    new_records = []
    for record in records_list:
        new_record = recompute_pectin_concentrations(
            old_record=record,
            particle_th_concentration=config["particle_concentration"],
            pectin_th_concentration=config["pectin_concentration"],
            calcium_chloride_concentration=config["calcium_chloride_concentration"],
        )
        new_records.append(new_record)

    # fmt: off
    capsules_records_values = [
        {k: v.value for k, v in record.items()}
        for record in new_records
    ]
    # fmt: on

    df_capsules_records = pd.DataFrame.from_records(capsules_records_values)

    return (
        new_records,
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


def plot_capsules_stats(
    # datapath: str | Path,
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

    scatterplot_num, scatterplot_vol = capsule_scatter(df_records=df_capsules_records)

    # hist.savefig(fname=get_save_path(datapath, "hist_kde", "png"))
    # kde.savefig(fname=get_save_path(datapath, "kde", "png"))
    # scatterplot.savefig(fname=get_save_path(datapath, "scatter", "png"))

    # plt.show()
    return hist, kde, scatterplot_num, scatterplot_vol


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
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Computes again the pectin concentrations without "
        "analysing the datafiles again. "
        "Incompatible with flag '--convert_to_8bit'.",
    )
    parser.add_argument(
        "--convert_to_8bit",
        action="store_true",
        help="Converts the input images to 8-bit images before processing. "
        "Incompatible with flag '--recompute'.",
    )
    args = parser.parse_args()
    if args.recompute and args.convert_to_8bit:
        # fmt: off
        raise ValueError(
            "Arguments '--recompute' and '--convert_to_8bit' "
            "are incompatible.",
        )
        # fmt: on

    if args.datapath is None:
        args.datapath = Path(askopenfilename())

    datapath, *_ = check_config(args.datapath)
    if args.recompute:
        (
            capsules_records,
            df_capsules_records,
            # capsules_processing_results,
        ) = remake_records(path=datapath)
    else:
        (
            capsules_records,
            df_capsules_records,
            # capsules_processing_results,
        ) = measure_capsules(
            datapath=datapath,
            reader=args.reader,
            convert_to_8bit=args.convert_to_8bit,
        )

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
