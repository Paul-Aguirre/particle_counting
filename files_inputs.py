from pathlib import Path
from typing import Sequence
from tkinter.filedialog import askopenfilenames

import numpy as np
from nd2reader import ND2Reader
import toml


def load_image_stack(
    path: str,
    start: int = 0,
    stop: int | None = None,
) -> tuple[np.ndarray, dict]:

    with ND2Reader(path) as images:
        image_stack = np.array([frame for frame in images])
        metadata = images.metadata

    image_stack = image_stack[start:stop]
    metadata["z_coordinates"] = metadata["z_coordinates"][start:stop]
    if stop:
        metadata["z_levels"] = range(start, stop)
    else:
        metadata["z_levels"] = range(start, metadata["z_levels"].stop)

    return image_stack, metadata


def initialize_toml(filename: Path | str, metadata: dict):
    toml_dict = {
        "stack_start": metadata["z_levels"].start,
        "stack_stop": metadata["z_levels"].stop,
        "particle_size_microns": 1,  # defaults to 1 µm
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
