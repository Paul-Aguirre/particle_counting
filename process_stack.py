import numpy as np
from skimage import filters, measure, morphology, draw


def process_stack(
    image_stack: np.ndarray,
    metadata: dict,
    particle_diameter: float,
    full_bbox: bool = True,
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

    props = measure.regionprops(
        labels,
        spacing=(
            np.mean(np.diff(np.array(metadata["z_coordinates"]))),  # z
            metadata["pixel_microns"],  # y
            metadata["pixel_microns"],  # x
        ),
    )

    bboxes3d = np.zeros(binary.shape, dtype=np.uint8)

    if full_bbox:
        for prop in props:
            minz, minr, minc, maxz, maxr, maxc = prop.bbox
            zz, rr, cc = draw.rectangle(
                start=(minz, minr, minc),
                end=(maxz, maxr, maxc),
                shape=binary.shape,
            )
            bboxes3d[zz, rr, cc] = 1
    else:
        for prop in props:
            minz, minr, minc, maxz, maxr, maxc = prop.bbox
            for z in range(minz, maxz):
                rr, cc = draw.rectangle_perimeter(
                    start=(minr, minc),
                    end=(maxr, maxc),
                    shape=binary.shape[1:],
                )
                bboxes3d[z, rr, cc] = 1

    results = {
        "otsu_threshold": thresh,
        "binary": binary,
        "binary_separated": binary_separated,
        "labels": labels,
        "props": props,
        "bboxes3d": bboxes3d,
    }

    return results
