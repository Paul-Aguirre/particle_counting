"""This module contains functions to initialise and help navigate this
configuration. The configuration is as follows:

./[datafile].nd2

./[datafile]/[datafile]_config.toml

./[datafile]/[datafile]_[result suffix].[ext]

This module also contains many helper functions to load and save data 
from different file type. The files are saved and imported under 
the assumption that a certain directory configuration is respected.
The supported file type are the following:

- ND2 data files
- TOML config files
- TOML result files
- CSV result files (exportation only)
"""

from pathlib import Path
from pprint import pprint
import tomllib
from typing import Generator, Sequence, Literal
from tkinter.filedialog import askopenfilenames
from typing import NamedTuple
import warnings

import numpy as np
from nd2reader import ND2Reader
import nd2
import pandas as pd
import toml


class Result(NamedTuple):
    """A NamedTuple representing a computation result."""

    value: float
    """The value of the result."""
    unit: str
    """The unit of the result."""


type ResultRecord = dict[Result]
"""A collection of results"""


def load_image_stack(
    path: str | Path,
    xstart: int = 0,
    xstop: int | None = None,
    ystart: int = 0,
    ystop: int | None = None,
    zstart: int = 0,
    zstop: int | None = None,
    zstep: int = 1,
    reader: Literal["nd2reader", "nd2"] = None,
) -> tuple[np.ndarray, dict]:
    """Loads the data and metadata from an ND2 file. Optionnaly only
    within specifyied selection boundaries, if specifyied, the metadata
    is corrected with the boundaries values. The boundaries are used for
    slicing the data in the image stack.

    Args:
        path (str | Path): A path-like object that points to the file to load.
        xstart (int, optional): Lower selection boundary over the x-axis.
            Defaults to 0.
        xstop (int | None, optional): Higher selection boundary over the
            x-axis. If None, data is sliced to the end of the x-axis.
            Defaults to None.
        ystart (int, optional): Lower selection boundary over the y-axis.
            Defaults to 0.
        ystop (int | None, optional): Higher selection boundary over the
            y-axis. If None, data is sliced to the end of the y-axis.
            Defaults to None.
        zstart (int, optional): Lower selection boundary over the z-axis.
            Defaults to 0.
        zstop (int | None, optional): Higher selection boundary over the
            z-axis. If None, data is sliced to the end of the z-axis.
            Defaults to None.
        zstep (int, optionnal): The slice step to select the images that
            will be loaded. Defaults to 1.

    Returns:
        tuple[np.ndarray, dict]: the image stack as a Numpy array and
            the metadata, corrected according to the selection
            parameters.
    """

    if reader is None or reader == "nd2reader":
        with ND2Reader(str(path)) as images:
            # fmt: off
            image_stack = np.array([
                frame for frame in images[zstart:zstop:zstep]
                ])
            # fmt: on
            metadata = images.metadata

    elif reader == "nd2":
        image_stack = nd2.imread(path)[zstart:zstop:zstep]
        with ND2Reader(str(path)) as images:
            metadata = images.metadata

    else:
        # fmt: off
        raise ValueError("`reader` argument should have values in "
                         "['nd2reader', 'nd2']")
        # fmt: on

    image_stack = image_stack[:, ystart:ystop, xstart:xstop]
    if xstop:
        metadata["width"] = xstop - xstart
    else:
        metadata["width"] = image_stack.shape[2] - xstart

    if ystop:
        metadata["height"] = ystop - ystart
    else:
        metadata["height"] = image_stack.shape[1] - ystart

    metadata["z_coordinates"] = metadata["z_coordinates"][zstart:zstop:zstep]
    if zstop:
        metadata["z_levels"] = range(zstart, zstop, zstep)
    else:
        metadata["z_levels"] = range(zstart, metadata["z_levels"].stop, zstep)

    return image_stack, metadata


def get_selection_stack(
    config: dict,
    datapath: str | Path,
    reader: Literal["nd2reader", "nd2"] = None,
) -> Generator[tuple[np.ndarray, dict], None, None]:
    """Iterator that yields substacks one by one. Substacks are selected
    using a list contained in the configuration file associted
    to the data file. The list can be found under the "selections" keyword.

    Args:
        config (dict): The data dictionnary loaded from the
            configuration file.
        datapath (str | Path): Path-like object pointing to the datafile.

    Yields:
        tuple[np.ndarray, dict]: The selected substack and corrected
            metadata corresponding to it.
    """
    for selection in config["selections"]:
        image_stack, metadata = load_image_stack(
            path=datapath,
            xstart=selection["xstart"],
            xstop=selection["xstop"],
            ystart=selection["ystart"],
            ystop=selection["ystop"],
            zstart=config["zstart"],
            zstop=config["zstop"],
            zstep=config["zstep"],
            reader=reader,
        )
        yield image_stack, metadata


def get_metadata(path: str | Path) -> dict:
    with ND2Reader(str(path)) as images:
        metadata: dict = images.metadata
    return metadata


def pprint_metadata(path: Path | str) -> None:
    metadata = get_metadata(path)
    metadata["z_levels"] = metadata["z_levels"].stop
    metadata["zstep_microns"] = np.mean(
        np.diff(
            np.array(
                metadata["z_coordinates"],
            )
        )
    )
    match metadata["pixel_microns"]:
        case 0.325:
            metadata["objective"] = "20x"
        case _:
            metadata["objective"] = "not 20x"

    not_printed_keys: list[str] = [
        "fields_of_view",
        "frames",
        "z_coordinates",
        "total_images_per_channel",
        "channels",
        "num_frames",
        "experiment",
        "events",
    ]
    for key in not_printed_keys:
        del metadata[key]

    pprint(metadata)


def initialize_config(filename: Path | str, metadata: dict) -> None:
    """Initializes TOML configuration file.

    Args:
        filename (Path | str): A path-like object pointing to where
            the configuration file will be created.
        metadata (dict): The metadata dictionnary of the corresponding
            data file.
    """
    toml_dict = {
        "zstart": metadata["z_levels"].start,
        "zstop": metadata["z_levels"].stop,
        "zstep": 1,
        "threshod_by_image": False,
        "particle_size_microns": 1,  # defaults to 1 µm
        "particle_concentration": 1e-4,  # in particles/µm^3
        "pectin_concentration": 40,  # in g/L
        "selections": [],
    }
    with open(filename, mode="w") as f:
        toml.dump(toml_dict, f)


def check_config(
    datapath: str | Path,
    make_pause: bool = False,
    show_warning: bool = False,
) -> tuple[Path, Path, Path]:
    """Given a datafile, checks that its associated directory and
    configuration TOML file exist and creates them otherwise.
    Also returns the three paths; the function is mainly used for that
    purpose in fact.

    Args:
        datapath (str | Path): Path-like object pointing to the data file.
        make_pause (bool, optional): If True, pauses the execution
            of the program to allow the user to edit the configuration
            file. Program execution is resumed when ENTER is pressed
            when in the terminal. Defaults to False.

    Returns:
        tuple[Path, Path, Path]: Paths to the datafile,
            the corresponding directory and configuration file,
            respectively.
    """
    datapath = Path(datapath)
    dirpath: Path = datapath.parent / f"{datapath.stem}"
    configpath: Path = dirpath / f"{datapath.stem}_config.toml"

    if not dirpath.exists():
        dirpath.mkdir()
        print(f"Created directory '{dirpath}'.")

    if not configpath.exists():
        metadata = get_metadata(datapath)
        initialize_config(configpath, metadata)
        print(f"Created file '{configpath}'.")
        if show_warning:
            warnings.warn(
                "You might want to edit the configuration file before"
                "you run further analysis on it.",
            )
        if make_pause:
            print("Please edit the configuration file.")
            input("Press Enter to continue.")

    return datapath, dirpath, configpath


def prepare_datafile(file: str | Path | Sequence[str | Path]) -> None:
    """Calls check_config() on one or more datafiles with the make_pause
    flag set to True.

    Args:
        file (str | Path | Sequence[str  |  Path]): Path-like object or
            sequence of them on which check_config() will be called.
    """
    if type(file) is str or type(file) is Path:
        check_config(file, make_pause=True)
    else:
        for elt in file:
            check_config(elt, make_pause=True)
    print("Configuration completed.")


def get_save_path(datapath: str | Path, suffix: str, ext: str) -> Path:
    """Returns a path with apporpriate name and location with regards
    to a data file to store additional data, results, etc.
    The path is customized through the suffix and ext arguments.

    Args:
        datapath (str | Path): Path-like object that points to the data
            file the returned path will be related to.
        suffix (str): Suffix to the returned path.
        ext (str): Extension to the returned path.

    Returns:
        Path: A Path object in the form "[datapath name]_[suffix].[ext]"
            located in the directory related to the data file.
    """
    datapath, dirpath, _ = check_config(datapath)
    save_path = dirpath / f"{datapath.stem}_{suffix}.{ext}"
    return save_path


def save_records(
    records_lst: list[ResultRecord],
    datapath: str | Path,
    suffix: str,
) -> Path:
    """Save result records in a toml file at the location appropriate
    to the specified datapath.

    Args:
        records_lst (list[ResultRecord]): List of ResultRecord objects
            to be saved.
        datapath (str | Path): The datapath to which the data is related.
        suffix (str): Suffix to the save file.

    Returns:
        Path: The path to the file in which the data was saved.
    """
    save_path = get_save_path(datapath, suffix, "toml")
    # copying results
    # results_lst_copy = [{k: v for k, v in d.copy().items()} for d in results_lst.copy()]
    # for results_lst in results_lst_copy:
    #     for key, result in results_lst.items():
    #         results_lst[key] = Result(float(result.value), result.unit)
    with open(save_path, "w") as f:
        # toml.dump({"results_lst": results_lst_copy}, f)
        toml.dump({"records_lst": records_lst}, f)
    return save_path


def load_records(path: str | Path):
    """Loads result data from a TOML file in the form of a list of
    ResultRecord objects.

    Args:
        path (str | Path): Path-like object pointing to the file to be
            loaded.

    Returns:
        list[ResultRecord]: List of ResultRecord objects.
    """
    with open(path, "r") as f:
        results_lst = toml.load(f)["records_lst"]
    for results_dict in results_lst:
        for key, result in results_dict.items():
            results_dict[key] = Result(eval(result[0]), result[1])
    return results_lst


def df_to_csv(
    df_records: pd.DataFrame,
    datapath: str | Path,
    suffix: str,
) -> Path:
    """Saves data records from a pandas.DataFrame to a CSV file at the
    location appropriate to the specified datapath.

    Args:
        df_records (pd.DataFrame): DataFrame containing measurement
            result records to be saved.
        datapath (str | Path): The datapath to which the data is related.
        suffix (str): Suffix to the save file.

    Returns:
        Path: The path to the file in which the data was saved.
    """
    save_path = get_save_path(datapath, suffix, "csv")
    df_records.to_csv(save_path)
    return save_path


def collect_results_from_csvs(
    datafiles: list[str | Path],
    suffix: str,
) -> pd.DataFrame:
    results_files: list[Path] = [
        get_save_path(datafile, suffix, "csv") for datafile in datafiles
    ]
    configs: list[dict] = []
    metadata_list: list[dict] = []
    for datafile in datafiles:
        *_, configpath = check_config(datapath=datafile)
        with open(configpath, "rb") as f:
            configs.append(tomllib.load(f))

        selected_metadata = {
            k: v
            for k, v in get_metadata(datafile).items()
            if k
            in [
                "pixel_microns",
                "width",
                "height",
                "z_levels",
                "z_coordinates",
            ]
        }
        selected_metadata["zstep_microns"] = float(
            np.median(np.diff(np.array(selected_metadata.pop("z_coordinates"))))
        )
        # fmt: off
        selected_metadata["num_steps"] = (
            selected_metadata["z_levels"].stop
            - selected_metadata["z_levels"].start
        )
        # fmt: on
        del selected_metadata["z_levels"]
        metadata_list.append(selected_metadata)

    dfs: list[pd.DataFrame] = [
        pd.concat(
            [
                pd.read_csv(results_file, header=0, index_col=0),
                pd.DataFrame.from_records(config),
                pd.DataFrame.from_records([metadata]),
            ],
            axis=1,
        )
        for (
            results_file,
            config,
            metadata,
        ) in zip(
            results_files,
            configs,
            metadata_list,
        )
    ]
    df_results = dfs[0]
    for df in dfs[1:]:
        df_results = pd.concat([df_results, df])
    df_results.reset_index(inplace=True, drop=True)

    return df_results


def main() -> None:
    # allows to prepares a list of selected files for analysis
    datafiles = askopenfilenames(title="Select the datafile to prepare")
    prepare_datafile(datafiles)
    # df_results = collect_results_from_csvs(datafiles, "capsule_records")
    # return df_results


if __name__ == "__main__":
    main()
