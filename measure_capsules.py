import tomllib
from pathlib import Path
from tkinter.filedialog import askopenfilename

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import toml

# from memory_profiler import profile

from files_io import (
    Result,
    ResultRecord,
    check_config,
    df_to_csv,
    get_save_path,
    get_selection_stack,
    save_records,
)
from process_stack import process_stack
from statistics_plots import capsule_distrib_kde, capsule_scatter


# @profile
def measure_capsules(datapath: str | Path) -> list[ResultRecord]:

    print(f"Analysing '{str(datapath)}'.")

    # * getting back the selections from the config file
    datapath, dirpath, configpath = check_config(datapath)
    with open(configpath, "rb") as f:
        config = tomllib.load(f)

    # * loading the substack from the main large image
    capsules_records = []
    # capsules_processing_results = []
    results = {}
    for substack, metadata in get_selection_stack(config=config, datapath=datapath):

        # * applying process stack to each substack
        process_stack(
            image_stack=substack,
            metadata=metadata,
            results=results,
            threshold=config.get("threshold"),
            morph_open=False,
            full_bboxes=False,
            capsule=True,
            use_centroids=True,
        )
        if not config.get("threshold"):
            config["threshold"] = int(results["threshold"])
            with open(configpath, "w") as f:
                toml.dump(config, f)
        # capsules_processing_results.append(results)

        # * extract capsule size parameters
        assert len(results["hull_props"]) == 1
        capsule_volume = results["hull_props"][0].area  # in µm^3
        # (area for 3D object is volume, checked in scikit image source code)
        capsule_diameter = results["hull_props"][0].equivalent_diameter_area
        # (also checked in skimage source code that formula applies to 3D case)

        # * and compute pectin concentration in capsule volume
        # two methods have been thought of for accessing only the tracers
        # in the hull:
        # - the die cutting approach:
        # particles_in_hull = results["hull"] == results["binary"]

        # - the centroid position check approach:
        particles_in_hull = np.zeros(substack.shape)

        for trac in results["props"]:
            # testing if tracer centroid is in the hull
            centroid_coords = tuple(int(coord) for coord in trac.centroid)
            if results["hull"][centroid_coords]:
                # extracting bbox dimensions/position to fill tracer
                # image back in it
                mind, minr, minc, maxd, maxr, maxc = trac.bbox
                depths = np.arange(mind, maxd)
                rows = np.arange(minr, maxr)
                cols = np.arange(minc, maxc)
                dgrid, rgrid, cgrid = np.meshgrid(
                    depths,
                    rows,
                    cols,
                    indexing="ij",
                )
                particles_in_hull[dgrid, rgrid, cgrid] = trac.image

        # the actual computations:
        particles_in_hull_results = process_stack(
            image_stack=particles_in_hull,
            metadata=metadata,
            binary=True,
            morph_open=False,
        )
        del particles_in_hull

        """ # ? Should we only count the volume of tracers inside
        the hull (current implementation)?
        Or instead count the volume of tracers which have their centroid
        in the hull ?
        Current implementation cuts some capsule volume on hull border
        thus reducing particle count.
        """

        # fmt: off
        median_num_pixels = np.median(
            np.array(
                [prop.num_pixels 
                 for prop in particles_in_hull_results["props"]],
            )
        )
        # fmt: on

        nums_pixels = np.array(
            [prop.num_pixels for prop in particles_in_hull_results["props"]]
        )

        num_particles_in_volume = np.sum(nums_pixels) / median_num_pixels

        # fmt: off
        particle_concentration_in_volume = (
            num_particles_in_volume / capsule_volume
            )
        # fmt: on

        pectin_experimental_concentration = (
            particle_concentration_in_volume
            * config["pectin_concentration"]
            / config["particle_concentration"]
        )

        capsule_record = {
            "capsule_diameter": Result(capsule_diameter, "um"),
            "capsule_volume": Result(capsule_volume, "um^3"),
            "median_num_pixels": Result(median_num_pixels, "pixels"),
            "num_particles_in_volume": Result(
                num_particles_in_volume,
                "particles",
            ),
            "particle_concentration_in_volume": Result(
                particle_concentration_in_volume, "particles/um^3"
            ),
            "pectin_experimental_concentration": Result(
                pectin_experimental_concentration, "g/L"
            ),
        }

        capsules_records.append(capsule_record)

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


def plot_save_capsule_stats() -> None:
    datapath, dirpath, configpath = check_config(askopenfilename())
    (
        capsules_records,
        df_capsules_records,
        # capsules_processing_results,
    ) = measure_capsules(datapath)

    print_capsule_records(capsules_records)
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
    return (
        capsules_records,
        df_capsules_records,
        # capsules_processing_results,
    )


if __name__ == "__main__":
    (
        capsules_records,
        df_capsules_records,
        # capsules_processing_results,
    ) = plot_save_capsule_stats()
