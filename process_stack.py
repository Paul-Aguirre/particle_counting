"""_summary_"""

import numpy as np

from plane import Plane
from process_stack_utils import (
    threshold_stack,
    morphological_opening,
    measure_particles,
    create_bboxes_stack,
    get_spacing_correction,
    make_centroids_stack,
    process_capsule_hull,
)


def process_stack(
    image_stack: np.ndarray,
    metadata: dict,
    results: dict = {},
    binary: bool = False,
    threshold: int | None = None,
    morph_open: bool = True,
    particle_diameter_um: float | None = None,  # in microns
    full_bboxes: bool = True,
    spacing: tuple[float, float, float] | None = None,
    capsule: bool = False,
    use_centroids: bool = False,
    filtering_plane: Plane = Plane.XY,
) -> dict:

    # * Isolating the particles by thresholding
    if not binary:
        if threshold:
            thresh, processed_stack = threshold_stack(
                image_stack=image_stack,
                thresh=threshold,
            )
        else:
            thresh, processed_stack = threshold_stack(image_stack)
        results.update(threshold=thresh, binary=processed_stack)
    else:
        processed_stack = image_stack

    if morph_open:
        processed_stack = morphological_opening(
            processed_stack=processed_stack,
            metadata=metadata,
            particle_diameter_um=particle_diameter_um,
        )
    results.update(binary_opened=processed_stack)

    # * Labeling and measuring particles properties
    labels, props, scaled_props = measure_particles(
        processed_stack=processed_stack,
        metadata=metadata,
    )
    results.update(labels=labels, props=props, scaled_props=scaled_props)

    # * Creating a stack of bboxes for displaying
    bboxes3d = create_bboxes_stack(
        stack_shape=processed_stack.shape,
        props=props,
        full_bboxes=full_bboxes,
    )
    results.update(bboxes3d=bboxes3d)

    # * Correcting spacing (optinnal) in order to make the tracers
    # * appear as spheres
    if spacing:
        spacing_corrected_props = get_spacing_correction(
            labels=labels,
            spacing=spacing,
        )
        results.update(spacing_corrected_props=spacing_corrected_props)

    # * Extracting the centroids
    if use_centroids:
        processed_stack = make_centroids_stack(
            props=props,
            shape=processed_stack.shape,
        )
        results.update(centroids_stack=processed_stack)

    # * Determine the convex hull (applicable for capsules)
    # (usable to determine capsule volume)
    if capsule:
        hull, hull_props = process_capsule_hull(
            processed_stack=processed_stack,
            props=props,
            metadata=metadata,
            plane=filtering_plane,
        )
        results.update(hull=hull, hull_props=hull_props)

    return results
