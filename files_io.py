from pathlib import Path
from typing import Sequence
from tkinter.filedialog import askopenfilenames

import numpy as np
from nd2reader import ND2Reader
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


def initialize_toml(filename: Path | str, metadata: dict):
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
        initialize_toml(configpath, metadata)
        print(f"Created {configpath.name}.")
        if make_pause:
            print("Please edit the configuration file.")
            input("Press Enter to continue.")

    return datapath, dirpath, configpath


def prepare_datafile(file: str | Path | Sequence[str | Path]):
    if type(file) is str or type(file) is Path:
        check_config(file)
    else:
        for elt in file:
            check_config(elt)


if __name__ == "__main__":
    # allows to prepares a list of selected files for analysis
    datafiles = askopenfilenames(title="Select the datafile to prepare")
    prepare_datafile(datafiles)
