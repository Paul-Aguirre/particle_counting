from pathlib import Path
from typing import Sequence
from tkinter.filedialog import askopenfilenames
from typing import NamedTuple

import numpy as np
from nd2reader import ND2Reader
import pandas as pd
import toml


def load_image_stack(
    path: str | Path,
    xstart: int = 0,
    xstop: int | None = None,
    ystart: int = 0,
    ystop: int | None = None,
    zstart: int = 0,
    zstop: int | None = None,
) -> tuple[np.ndarray, dict]:

    with ND2Reader(str(path)) as images:
        image_stack = np.array([frame for frame in images])
        metadata = images.metadata

    image_stack = image_stack[zstart:zstop, ystart:ystop, xstart:xstop]
    if xstop:
        metadata["width"] = xstop - xstart
    else:
        metadata["width"] = image_stack.shape[2] - xstart

    if ystop:
        metadata["height"] = ystop - ystart
    else:
        metadata["height"] = image_stack.shape[1] - ystart

    metadata["z_coordinates"] = metadata["z_coordinates"][zstart:zstop]
    if zstop:
        metadata["z_levels"] = range(zstart, zstop)
    else:
        metadata["z_levels"] = range(zstart, metadata["z_levels"].stop)

    return image_stack, metadata


def get_selection_stack(config: dict, path: str | Path):
    for selection in config["selections"]:
        image_stack, metadata = load_image_stack(
            path=path,
            xstart=selection["xstart"],
            xstop=selection["xstop"],
            ystart=selection["ystart"],
            ystop=selection["ystop"],
            zstart=config["zstart"],
            zstop=config["zstop"],
        )
        yield image_stack, metadata


def initialize_config(filename: Path | str, metadata: dict):
    toml_dict = {
        "zstart": metadata["z_levels"].start,
        "zstop": metadata["z_levels"].stop,
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
):
    datapath = Path(datapath)
    dirpath = datapath.parent / f"{datapath.stem}"
    configpath = dirpath / f"{datapath.stem}_config.toml"

    if not dirpath.exists():
        dirpath.mkdir()
        print(f"Created directory '{dirpath}'.")

    if not configpath.exists():
        _, metadata = load_image_stack(str(datapath))
        initialize_config(configpath, metadata)
        print(f"Created {configpath.name}.")
        if make_pause:
            print("Please edit the configuration file.")
            input("Press Enter to continue.")

    return datapath, dirpath, configpath


def prepare_datafile(file: str | Path | Sequence[str | Path]):
    if type(file) is str or type(file) is Path:
        check_config(file, make_pause=True)
    else:
        for elt in file:
            check_config(elt, make_pause=True)


class Result(NamedTuple):
    value: float
    unit: str


def get_save_path(datapath: str | Path, suffix: str, ext: str):
    datapath, dirpath, _ = check_config(datapath)
    save_path = dirpath / f"{datapath.stem}_{suffix}.{ext}"
    return save_path


def save_records(
    records_lst: list[dict],
    datapath: str | Path,
    suffix: str,
) -> Path:
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
):
    save_path = get_save_path(datapath, suffix, "csv")
    df_records.to_csv(save_path)
    return save_path


if __name__ == "__main__":
    # allows to prepares a list of selected files for analysis
    datafiles = askopenfilenames(title="Select the datafile to prepare")
    prepare_datafile(datafiles)
