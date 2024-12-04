"""This module contains only a convinience function to perform various
processing operations on image stacks. All the othter functions used by
this function are regrouped in a separate module: process_stack_utils.py
"""

import numpy as np
import pandas as pd
from skimage import measure

# from memory_profiler import profile

from plane import Plane
from process_stack_utils import (
    threshold_images_in_stack,
    threshold_stack,
    morphological_opening,
    # measure_particles,
    create_bboxes_stack,
    make_centroids_stack,
    process_capsule_hull,
)


# @profile
def process_stack(
    image_stack: np.ndarray,
    metadata: dict,
    results: dict | None = None,
    binary: bool = False,
    threshold: int | None = None,
    threshold_by_image: bool = False,
    morph_open: bool = False,
    particle_diameter_um: float | None = None,  # in microns
    scale_props: bool = False,
    bboxes: bool = False,
    full_bboxes: bool = False,
    # spacing: tuple[float, float, float] | None = None, # deprecated
    capsule: bool = False,
    use_centroids: bool = True,
    trac_nb_thresh: int = 20,
    filtering_plane: Plane = Plane.XY,
) -> dict:
    """A convinience function to process an image stack. The different
    processing steps are enabled or disabled through flag arguments.
    The order of the flags in the functions signature is the same as the
    order in which the processing step are performed.
    This function is used to process both reference sample and capsule
    image stacks.

    Args:
        image_stack (np.ndarray): An image stack, grayscale or binary.
        metadata (dict): The metadata associated to the image stack.
            It is required to have specific keys:

            * "pixel_microns" that contains the pixel size expressed in
                microns
            * "z_coordinates" that contains a list for the z coordinates
                of the individual images in the stack

        results (dict, optional): A result dictionnary similar to
            the one returned by this function. The results of
            the processing steps are stored in this dictionnary.
            An existing keys will be overwritten if the corresponding
            processing step is requested. Defaults to {}.
        binary (bool, optional): If True, the image is thresholded.
            Defaults to False.
        threshold (int | None, optional): Used a threshold value if
            provided. Otsu's method is used to determine the threshold
            otherwise. Defaults to None.
        morph_open (bool, optional): If True, a morphological opening
            opening operation is performed on the image stack.
            Defaults to False.
        particle_diameter_um (float | None, optional): The particle
            diameter used in the analysed sample. Only necessary if
            morph_open is set to True, will be ignored otherwise.
            Defaults to None.
        scale_props (bool, optionnal): If True, computes the region
            properties a second time while taking the x, y and z
            resolution of the image stack. The list of these properties
            is stored at the "scaled_props" in the results dictionnary.
            Defaults to False.
        bboxes (bool, optionnal): If True, creates a binary image stack
            of the same shape as `image_stack` containing the bboxes of
            the elements in props. This image stack is stored at
            the "bboxes3d" key in the results dictionnary.
        full_bboxes (bool, optionnal): If True, the bboxes returned at
            the "bboxes3d" key are full, and thus appear as filled
            squares when plotted in 2D. If False, the bboxes are drawn
            as square perimeters when plotted in 2D, this prevents
            overlapping when displaying. Ignored if `bboxes` flag is set
            to False. Defaults to False.
        spacing (tuple[float, float, float] | None, optional): A (z,y,x)
            tuple. Is used, if provided, to compute the region
            properties of the tracers again. The results are listed
            separately from the original region properties.
            Defaults to None. (Deprecated)
        capsule (bool, optional): If True, the capsule volume is
            isolated as a convex hull and its properties are measured.
            The images in the stack are first filtered by number of
            tracers before computing the convex hull to eliminate lonely
            suspended tracers. Defaults to False.
        use_centroids (bool, optional): If True, an image stack is
            created containing only tracers centroids. Using this image
            stack instead of full tracers images for the convex hull
            calculation significantly reduces computation time.
            Defaults to True.
        trac_nb_thresh (int, optional): The number of tracer to be used
            as threshold for starting to include images in the convex
            hull computation. This will not result in not counting
            images in the middle of the capsule. Defaults to 20.
        filtering_plane (Plane, optional): The plane type in which
            to perform the number thresholding step. Defaults to Plane.XY.

    Returns:
        dict: Contains the results of all performed operations stored at
            specific keys. Here is the list of possible keys and
            the values associated to them:

            * "threshold": the intensity threshold value used to obtain
                the binary image stack
            * "binary": binary image stack
            * "opened": morphologically opened binary image stack
            * "labels": labeled binary image stack
            * "props": list of region properties for all labeled regions
                present in image stack stored at the "labels" key
            * "scaled_props": list of region properties computed with
                accurate scaling for all labeled regions present in
                image stack stored at the "labels" key
            * "bboxes3d": binary image stack containing the bboxes for
                the measured regions/tracers.
            * "spacing_corrected_props": list of region properties for
                all labeled regions present image stack computed with
                provided spacing tuple stored at the "labels" key (deprecated)
            * "centroids_stack": binary image stack containing centroids
                positions for the regions at "labels"
            * "hull": binary image stack of the convex hull
            * "hull_props": list of region properties (of length 1) for
                the convex hull at "hull"
    """
    if results is None:
        results = {}  # ensuring a fresh empty dictionnary

    # * Isolating the particles by thresholding
    if not binary:
        if threshold:
            threshold, processed_stack = threshold_stack(
                image_stack=image_stack,
                threshold=threshold,
            )
        elif threshold_by_image:
            threshold, processed_stack = threshold_images_in_stack(
                image_stack=image_stack,
            )
        else:
            threshold, processed_stack = threshold_stack(image_stack)
        results.update(threshold=threshold, binary=processed_stack)
    else:
        processed_stack = image_stack

    # * Trying to separate particles by morphological opening
    if morph_open:
        processed_stack = morphological_opening(
            processed_stack=processed_stack,
            metadata=metadata,
            particle_diameter_um=particle_diameter_um,
        )
        results.update(opened=processed_stack)

    # * Labeling and measuring particles properties
    labels = measure.label(processed_stack).astype(np.uint16)
    props = measure.regionprops(labels)
    df_props = pd.DataFrame(
        measure.regionprops_table(
            label_image=labels,
            properties=(
                "label",
                "bbox",
                "centroid",
                "area",
                "num_pixels",
                "slice",
                "equivalent_diameter_area",
            ),
        )
    )
    df_props.rename(
        columns={
            "bbox-0": "bbox_zmin",
            "bbox-1": "bbox_ymin",
            "bbox-2": "bbox_xmin",
            "bbox-3": "bbox_zmax",
            "bbox-4": "bbox_ymax",
            "bbox-5": "bbox_xmax",
            "centroid-0": "centroid_z",
            "centroid-1": "centroid_y",
            "centroid-2": "centroid_x",
        },
        inplace=True,
    )
    results.update(props=props, df_props=df_props)
    if scale_props:
        scaled_props = measure.regionprops(
            label_image=labels,
            spacing=(
                np.mean(np.diff(np.array(metadata["z_coodinates"]))),  # z
                metadata["pixel_microns"],  # y
                metadata["pixel_microns"],  # x
            ),
        )
        results.update(scaled_props=scaled_props)
    del labels

    # * Creating a stack of bboxes for displaying
    if bboxes:
        bboxes3d = create_bboxes_stack(
            stack_shape=processed_stack.shape,
            props=props,
            full_bboxes=full_bboxes,
        )
        results.update(bboxes3d=bboxes3d)

    # * Correcting spacing (optinnal) in order to make the tracers
    # * appear as spheres
    # if spacing:
    #     spacing_corrected_props = measure.regionprops(
    #         label_image=labels,
    #         spacing=spacing,
    #     )
    #     results.update(spacing_corrected_props=spacing_corrected_props)

    # * Extracting the centroids
    if use_centroids and capsule:
        processed_stack = make_centroids_stack(
            props=props,
            shape=processed_stack.shape,
        )
        results.update(centroids_stack=processed_stack)

    # * Determine the convex hull (applicable for capsules)
    # (usable to determine capsule volume)
    if capsule:
        hull, hull_props, hull_props_table = process_capsule_hull(
            processed_stack=processed_stack,
            props=props,
            metadata=metadata,
            trac_nb_thresh=trac_nb_thresh,
            plane=filtering_plane,
        )
        results.update(
            hull=hull,
            hull_props=hull_props,
            hull_props_table=hull_props_table,
        )

    return results
