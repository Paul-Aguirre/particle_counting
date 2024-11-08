import argparse
from pathlib import Path
from tkinter.filedialog import askopenfilename
import tomllib
from typing import Literal

from matplotlib import pyplot as plt
import pandas as pd

from files_io import (
    Result,
    ResultRecord,
    check_config,
    df_to_csv,
    get_selection_slices,
    save_records,
)
from measure_capsules import print_capsule_records
from process_stack import process_stack
from statistics_plots import particle_distributions


def measure_capsule_slices(
    datapath: Path | str,
    reader: Literal["nd2reader", "nd2"],
) -> tuple[list[ResultRecord], pd.DataFrame]:

    print(f"Analysing '{str(datapath)}'.")

    # * Loading configuration
    datapath, dirpath, configpath = check_config(datapath)
    with open(configpath, "rb") as f:
        config = tomllib.load(f)

    if not config["selections"]:
        raise ValueError("No selected region in the stack.")

    capsules_records: list = []
    results: dict
    n_iter: int = 1
    # * Loading selected images
    for image, metadata in get_selection_slices(
        config=config,
        datapath=datapath,
        reader=reader,
    ):
        # * Applying process_stack to each image
        results = process_stack(
            image_stack=image,
            metadata=metadata,
            binary=False,
            threshold=config.get("threshold"),
            threshold_by_image=False,
            capsule=True,
            use_centroids=True,
            trac_nb_thresh=0,
        )

        # * Plotting processing steps:
        # - initial image
        fig_initial, ax_initial = plt.subplots()
        ax_initial.imshow(
            image,
            cmap="grey",
            norm="log",
        )
        fig_initial.savefig(
            fname=dirpath / f"{datapath.stem}_initial_middle_no-{n_iter}.png",
            format="png",
        )

        # - binary
        fig_binary, ax_binary = plt.subplots()
        ax_binary.imshow(
            results["binary"],
            cmap="grey",
        )
        fig_binary.savefig(
            fname=dirpath / f"{datapath.stem}_binary_middle_no-{n_iter}.png",
            format="png",
        )

        # - initial + convex hull
        ax_initial.imshow(
            results["hull"],
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

        # * Extracting capsule size parameters
        capsule_area = results["hull_props"][0].area  # in um^2
        capsule_diameter = results["hull_props"][0].equivalent_diameter_area

        # * Creating record
        capsule_record = {
            "capsule_area": Result(capsule_area, "um^2"),
            "capsule_diameter": Result(capsule_diameter, "um"),
        }
        capsules_records.append(capsule_record)
        n_iter += 1

    # * Organising the results
    # fmt: off
    capsules_records_values = [
        {k: v.value for k, v in record.items()} 
        for record in capsules_records
    ]
    # fmt: on

    df_capsules_records = pd.DataFrame.from_records(capsules_records_values)
    df_capsules_records["c_0"] = config["pectin_concentration"]  # ?

    return (
        capsules_records,
        df_capsules_records,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="measure_capsule_slices")
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
    ) = measure_capsule_slices(datapath=datapath, reader=args.reader)

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
