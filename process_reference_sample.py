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


def compute_total_volume(metadata: dict) -> np.float64:

    total_volume = np.float64(
        abs(metadata["z_coordinates"][-1] - metadata["z_coordinates"][0])
        * metadata["width"]
        * metadata["height"]
        * metadata["pixel_microns"] ** 2
    )
    return total_volume


def compute_particle_concentration_in_number(
    processing_results: dict,
    total_volume: float,
) -> tuple[np.float64, np.float64]:

    num_regions = np.float64(len(processing_results["props"]))
    particle_concentration_in_number = np.float64(num_regions / total_volume)
    return num_regions, particle_concentration_in_number


def compute_particle_concentration_in_volume(
    processing_results: dict,
    total_volume: float,
) -> tuple[np.float64, np.float64, np.float64]:

    # fmt: off
    nums_pixels = np.array([
        prop.num_pixels for prop in processing_results["props"]
        ])
    # fmt: on
    median_num_pixels = np.median(nums_pixels)
    num_particles_in_volume = np.sum(nums_pixels) / median_num_pixels
    particle_concentration_in_volume = num_particles_in_volume / total_volume

    return (
        median_num_pixels,
        num_particles_in_volume,
        particle_concentration_in_volume,
    )


def process_reference_sample(
    datapath: str | Path,
    view_stack: bool = False,
    view_distributions: bool = True,
) -> tuple:

    # * Checking and loading configuration
    datapath, dirpath, configpath = check_config(datapath)

    with open(configpath, "rb") as f:
        config = tomllib.load(f)

    # * Loading image stack
    image_stack, metadata = load_image_stack(
        path=datapath,
        zstart=config["zstart"],
        zstop=config["zstop"],
        zstep=config["zstep"],
    )

    # * processing image stack
    if view_stack:
        results = process_stack(
            image_stack=image_stack,
            metadata=metadata,
            threshold=config.get("threshold"),  # to avoid recomputing it
            threshold_by_image=config.get("threshold_by_image", False),
            morph_open=False,
            # morph_open=True,
            particle_diameter_um=config["particle_size_microns"],
            bboxes=True,
            full_bboxes=False,
        )
    else:
        results = process_stack(
            image_stack=image_stack,
            metadata=metadata,
            threshold=config.get("threshold"),  # to avoid recomputing it
            threshold_by_image=config.get("threshold_by_image", False),
            morph_open=False,
            # morph_open=True,
            particle_diameter_um=config["particle_size_microns"],
            bboxes=False,
        )

    # to avoid recomputing threshold, it is saved
    if not config.get("threshold"):
        config["threshold"] = int(results["threshold"])
        with open(configpath, "w") as f:
            toml.dump(config, f)

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
    del image_stack  # for memory economy

    # * Computing particle concentration in number
    total_volume = compute_total_volume(metadata=metadata)

    (
        num_regions,
        particle_concentration_in_number,
    ) = compute_particle_concentration_in_number(
        processing_results=results,
        total_volume=total_volume,
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
            fname=dirpath / f"{datapath.stem}_{key}_hist.png",
            format="png",
            # transparent=True,
        )

    # * Computing particle concentration in volume
    # ! le nombre de pixel médian est trop élevé pour obtenir un calcul
    # ! correct de la concentration en particules en volume pour les
    # ! traceurs de 0.2 microns.
    # ? Utiliser le fractile d'ordre 0.1 ou 0.2 ?
    (
        median_num_pixels,
        num_particles_in_volume,
        particle_concentration_in_volume,
    ) = compute_particle_concentration_in_volume(
        processing_results=results,
        total_volume=total_volume,
    )

    if view_distributions:
        plt.show()

    # todo: add units and validation values from plots (medians)
    ref_results = {
        "zstep": Result(
            value=np.median(np.diff(np.array(metadata["z_coordinates"]))),
            unit="um",
        ),
        "total_volume": Result(total_volume, "um^3"),
        "num_particles_counted": Result(num_regions, "particles"),
        "num_particle_concentration": Result(
            value=particle_concentration_in_number,
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

    return metadata, results, ref_results, histograms, plots


def test_threshold_sensitivity(
    datapath: str | Path,
    threshold_variation: float = 0.1,
) -> None:
    print(f"Analysing '{str(datapath)}'")

    # * Checking and loading configuration
    datapath, dirpath, configpath = check_config(datapath)

    with open(configpath, "rb") as f:
        config = tomllib.load(f)

    # * Loading image stack
    image_stack, metadata = load_image_stack(
        path=datapath,
        zstart=config["zstart"],
        zstop=config["zstop"],
        zstep=config["zstep"],
    )
    total_volume = compute_total_volume(metadata=metadata)
    threshold: int = config["threshold"]
    thresholds: list = [
        round(threshold - threshold_variation * threshold),
        threshold,
        round(threshold + threshold_variation * threshold),
    ]
    for thresh in thresholds:
        results = process_stack(
            image_stack=image_stack,
            metadata=metadata,
            threshold=thresh,
            threshold_by_image=config.get("threshold_by_image", False),
            morph_open=True,
            particle_diameter_um=config["particle_size_microns"],
            bboxes=False,
        )
        (
            num_regions,
            particle_concentration_in_number,
        ) = compute_particle_concentration_in_number(
            processing_results=results,
            total_volume=total_volume,
        )
        (
            median_num_pixels,
            num_particles_in_volume,
            particle_concentration_in_volume,
        ) = compute_particle_concentration_in_volume(
            processing_results=results,
            total_volume=total_volume,
        )
        ref_results = {
            "threshold_value": Result(thresh, ""),
            "total_volume": Result(total_volume, "um^3"),
            "num_particles_counted": Result(num_regions, "particles"),
            "num_particle_concentration": Result(
                value=particle_concentration_in_number,
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
        print("")
        for key, value in ref_results.items():
            line_title = key.replace("_", " ").capitalize()
            print(f"{line_title}: {value.value:.4e} {value.unit}")
        print("")


def print_reference_results(datapath: Path, results: dict):
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


def main() -> None:
    datapath = Path(askopenfilename(title="Choose a data file"))
    if not datapath.is_file():
        raise ValueError("No file was selected.")
    (
        metadata,
        results,
        ref_results,
        histograms,
        plots,
    ) = process_reference_sample(
        datapath,
        view_stack=False,
        view_distributions=True,
    )
    print_reference_results(datapath, ref_results)
    save_results_path = save_records(
        [ref_results],
        datapath=datapath,
        suffix="reference_results",
    )
    print(f'Reference sample results saved at "{save_results_path}".')
    return (
        metadata,
        results,
        ref_results,
        histograms,
        plots,
    )


if __name__ == "__main__":
    (
        metadata,
        results,
        ref_results,
        histograms,
        plots,
    ) = main()
    # main()
    # test_threshold_sensitivity(askopenfilename())
