import tomllib
from pathlib import Path
from tkinter.filedialog import askopenfilename

from files_io import check_config, load_image_stack
from process_stack import process_stack


def measure_capsules(datapath: str | Path):
    # * getting back the selections from the config file
    with open(datapath, "rb") as f:
        config = tomllib.load(f)

    # * loading the substack from the main large image
    def get_selection_stack(config: dict, path: str | Path):  # reloading vesion
        # todo: check which is better
        # todo: - preloading the stack before the loop (time efficient)
        # todo: - reloading the stack at every iteration (supposedly memory efficient)
        for selection in config["selections"]:
            image_stack, metadata = load_image_stack(
                path=path,
                xstart=selection["xstart"],
                xstop=selection["xstop"],
                ystart=selection["ystart"],
                ystop=selection["ystop"],
                zstart=config["stack_start"],
                zstop=config["stack_stop"],
            )
            yield image_stack, metadata

    for substack, metadata in get_selection_stack(config=config, path=path):
        # * applying process stack to each substack
        results = process_stack(
            image_stack=substack,
            metadata=metadata,
            morph_open=False,
            full_bboxes=False,
            capsule=True,
            use_centroids=True,
        )
        # * extract capsule size
        capsule_size = results["hull_props"].area  # in µm^3
        # * and compute pectin concentration in capsule volume

    # * plot capsule size distribution
    # * plot pectin concentration in capsule against capsule radius


if __name__ == "__main__":
    path = Path(askopenfilename())
    measure_capsules(path)
