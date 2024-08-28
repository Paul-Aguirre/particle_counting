import numpy as np
from skimage import filters, measure, morphology, draw


def process_stack(
    image_stack: np.ndarray,
    metadata: dict,
    particle_diameter: float | None = None,  # in microns
    morph_open: bool = True,
    full_bbox: bool = True,
    spacing: tuple[float, float, float] | None = None,
    capsule: bool = False,
) -> dict:

    results = {}
    # Isolating the particles by thresholding
    thresh = filters.threshold_otsu(image_stack)
    processed_stack = image_stack > thresh

    results.update(ostu_threshold=thresh, binary=processed_stack)

    if morph_open:
        # Separating grouped particles for later counting
        structuring_element = morphology.ball(
            max(
                1, particle_diameter / metadata["pixel_microns"] / 2
            )  # radius in pixels
        )

        processed_stack = morphology.binary_opening(
            processed_stack, structuring_element
        )
        results.update(binary_opened=processed_stack)
    else:
        results.update(binary_opened=None)

    # Detecting and labeling particles
    labels = measure.label(processed_stack)
    results.update(labels=labels)

    props = measure.regionprops(
        labels,
        spacing=(
            np.mean(np.diff(np.array(metadata["z_coordinates"]))),  # z
            metadata["pixel_microns"],  # y
            metadata["pixel_microns"],  # x
        ),
    )
    results.update(props=props)

    bboxes3d = np.zeros(processed_stack.shape, dtype=np.uint8)

    if full_bbox:
        for prop in props:
            minz, minr, minc, maxz, maxr, maxc = prop.bbox
            zz, rr, cc = draw.rectangle(
                start=(minz, minr, minc),
                end=(maxz, maxr, maxc),
                shape=processed_stack.shape,
            )
            bboxes3d[zz, rr, cc] = 1
    else:
        for prop in props:
            minz, minr, minc, maxz, maxr, maxc = prop.bbox
            for z in range(minz, maxz):
                rr, cc = draw.rectangle_perimeter(
                    start=(minr, minc),
                    end=(maxr, maxc),
                    shape=processed_stack.shape[1:],
                )
                bboxes3d[z, rr, cc] = 1
    results.update(bboxes3d=bboxes3d)

    if spacing:
        spacing_corrected_props = measure.regionprops(
            labels,
            spacing=spacing,
        )
        results.update(spacing_corrected_props=spacing_corrected_props)
    else:
        results.update(spacing_corrected_props=None)

    if capsule:
        hull = morphology.convex_hull_image(processed_stack)
        labeled_hull = measure.label(hull)
        hull_props = measure.regionprops(
            labeled_hull,
            spacing=(
                np.mean(np.diff(np.array(metadata["z_coordinates"]))),  # z
                metadata["pixel_microns"],  # y
                metadata["pixel_microns"],  # x
            ),
        )
        results.update(hull=hull, hull_props=hull_props)
    else:
        results.update(hull=None, hull_props=None)

    return results
