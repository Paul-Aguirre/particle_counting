import numpy as np
from nd2reader import ND2Reader


def load_image_stack(path: str) -> tuple[np.ndarray, dict]:
    with ND2Reader(path) as images:
        image_stack = np.array([frame for frame in images])
        metadata = images.metadata

    return image_stack, metadata
