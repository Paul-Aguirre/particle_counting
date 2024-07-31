import numpy as np
from nd2reader import ND2Reader


def load_image_stack(
    path: str,
    start: int = 0,
    stop: int | None = None,
) -> tuple[np.ndarray, dict]:

    with ND2Reader(path) as images:
        image_stack = np.array([frame for frame in images])
        metadata = images.metadata

    image_stack = image_stack[start:stop]
    metadata["z_coodinates"] = metadata["z_coordinates"][start:stop]
    if stop:
        metadata["z_levels"] = range(start, stop)
    else:
        metadata["z_levels"] = range(start, metadata["z_levels"].stop)

    return image_stack, metadata
