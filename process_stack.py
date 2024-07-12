import numpy as np
from skimage import filters, measure, morphology


def process_stack(
    image_stack: np.ndarray,
    metadata: dict,
    particle_diameter: float,
) -> dict:
    # Isolating the particles by thresholding
    thresh = filters.threshold_otsu(image_stack)
    binary = image_stack > thresh

    # Separating grouped particles for later counting
    structuring_element = morphology.ball(
        particle_diameter / metadata["pixel_microns"] / 2  # radius in pixels
    )

    binary_separated = morphology.binary_opening(binary, structuring_element)

    # Detecting and labeling particles
    labels = measure.label(binary_separated)

    props = measure.regionprops(labels)

    results = {
        "otsu_threshold": thresh,
        "binary": binary,
        "binary_separated": binary_separated,
        "labels": labels,
        "props": props,
    }

    return results
