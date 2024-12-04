"""This module contains all the functions used in the process_stack()
function from the process_stack.py module.
"""

from pathlib import Path

import numpy as np
from skimage import filters, measure, morphology, draw
from nd2reader import ND2Reader
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_fill_holes

# from memory_profiler import profile

from plane import Plane, Region


# @profile
def threshold_stack(
    image_stack: np.ndarray,
    threshold: int | None = None,
) -> tuple[int, np.ndarray]:
    """Returns an image stack thresholded either by the provided thresh
    value or by the one determined with Otsu's method.

    Args:
        image_stack (np.ndarray): A grayscale image stack.
        thresh (int | None, optional): A threshold value. If left
            to None, the threshold is determined by otsu's method.
            Defaults to None.

    Returns:
        tuple[int, np.ndarray]: The threshold value and the binary image
            stack resulting from the thresholding.
    """
    if not threshold:
        threshold = filters.threshold_otsu(image_stack)
    processed_stack = image_stack > threshold
    return threshold, processed_stack


def threshold_images_in_stack(
    image_stack: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    """Calculates and applies the threshold for and to an image stack,
    image by image.

    Args:
        image_stack (np.ndarray): Image stack to be thresholded.

    Returns:
        tuple[list[int], np.ndarray]: list of the thresholds, thresholded stack.
            The thresholds are aranged in ascending z index with respect
            to the images.
    """
    thresholds: list = []
    processed_stack: np.ndarray = np.zeros(image_stack.shape, dtype=bool)
    for z in range(image_stack.shape[0]):
        thresholds.append(filters.threshold_otsu(image_stack[z]))
        processed_stack[z] = image_stack[z] > thresholds[-1]
    return thresholds, processed_stack


def threshold_images_in_file(datapath: str | Path) -> list[int]:
    # thougth of for memory economy.
    thresholds: list = []
    with ND2Reader(str(datapath)) as images:
        for image in images:
            thresholds.append(filters.threshold_otsu(image))
    return thresholds


# @profile
def morphological_opening(
    processed_stack: np.ndarray,
    metadata: dict,
    particle_diameter_um: float,
) -> np.ndarray:
    """Performs a morphological opening on a binary image stack using
    a ball structuring element with same radius as the tracers used in
    the sample or of at least one pixel (does not change the image).

    Args:
        processed_stack (np.ndarray): A binary image stack to with
        overlapping regions.
        metadata (dict): The metadata associated with the image stack.
        particle_diameter_um (float): The diameter of the tracer in
        the sample.

    Returns:
        np.ndarray: The image stack after the morphological opening.
    """
    structuring_element = morphology.ball(  # radius in pixels:
        max(1, particle_diameter_um / metadata["pixel_microns"] / 2)
    )
    # The max() function here ensure that the structuring element is
    # at least 1 pixel otherwise the result of the opening is
    # a black image.

    processed_stack = morphology.binary_opening(
        image=processed_stack,
        footprint=structuring_element,
    )
    return processed_stack


# @profile
def measure_particles(
    processed_stack: np.ndarray,
    metadata: dict,
) -> tuple:
    """Labels regions and measures region properties on an binary image
    stack.

    Args:
        processed_stack (np.ndarray): A binary image stack to with
            regions to measure.
        metadata (dict): The metadata associated to the image stack.

    Returns:
        tuple: The labeled image stack, a list of region properties,
            a list of region properties scaled with pixel size data
            present in the provided metadata.
    """
    labels = measure.label(processed_stack)

    props = measure.regionprops(labels)
    scaled_props = measure.regionprops(
        labels,
        spacing=(
            np.mean(np.diff(np.array(metadata["z_coordinates"]))),  # z
            metadata["pixel_microns"],  # y
            metadata["pixel_microns"],  # x
        ),
    )
    return props, scaled_props


# @profile
def create_bboxes_stack(
    stack_shape: tuple[int, int, int],
    props: list[Region],
    full_bboxes: bool = False,
) -> np.ndarray:
    """Returns a binary image stack of the given shape containing
    the bboxes of the elements in props.

    Args:
        stack_shape (tuple[int, int, int]): The shape of the output
            image stack.
        props (list[Region]): List of region properties which's bboxes
            will be added to the output stack.
        full_bboxes (bool, optional): If True, the bboxes will be drawn
            as full squares if plotted in 2D. If False, the bboxes will
            be drawn as empty square perimeters. Defaults to False.

    Returns:
        np.ndarray: An image stack containing the bboxes in
            the format specified with full_bboxes.
    """
    bboxes3d = np.zeros(stack_shape, dtype=np.uint8)
    if full_bboxes:
        for prop in props:
            minz, minr, minc, maxz, maxr, maxc = prop.bbox
            zz, rr, cc = draw.rectangle(
                start=(minz, minr, minc),
                end=(maxz, maxr, maxc),
                shape=stack_shape,
            )
            bboxes3d[zz, rr, cc] = 1
    else:
        for prop in props:
            minz, minr, minc, maxz, maxr, maxc = prop.bbox
            for z in range(minz, maxz):
                rr, cc = draw.rectangle_perimeter(
                    start=(minr, minc),
                    end=(maxr, maxc),
                    shape=stack_shape[1:],
                )
                bboxes3d[z, rr, cc] = 1
    return bboxes3d


# @profile
def make_centroids_stack(
    props: list[Region],
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Returns an binary image stack of specified shape containing True
    where the centroids of the elements in props are located.

    Args:
        props (list[Region]): List of region properties used to which's
            centroids will be placed in the output image stack.
        shape (tuple[int, int, int]): The shape of the ouput image
            stack.

    Returns:
        np.ndarray: The image stack containing the positions of
            the centroids.
    """

    centroids_coords = [prop.centroid for prop in props]
    centroids_stack = np.zeros(shape=shape, dtype=bool)
    for coords in centroids_coords:
        try:
            rounded_coords = tuple([int(coord.round()) for coord in coords])
            centroids_stack[rounded_coords] = True
        except IndexError:
            pass

    return centroids_stack


# @profile
def process_capsule_hull(
    processed_stack: np.ndarray,
    props: list[Region],
    metadata: dict,
    trac_nb_thresh: int,
    no_copy: bool = False,
    plane: Plane = Plane.XY,
) -> tuple:
    """Processes image stack by filtering the images by number of
    tracers, computing the convex hull drawn by the tracers and
    measuring the region properties of the hull.

    Args:
        processed_stack (np.ndarray): Image stack to be processed.
        props (list[Region]): List of region properties of the tracers
            in the image stack.
        metadata (dict): Metadata of the image stack.
        trac_nb_thresh (int): threshold for starting to include images
            in the convex hull computation.
        plane (Plane, optional): The plane type in which to perform
            the number thresholding step. Defaults to Plane.XY.

    Returns:
        tuple: An image stack of the computed convex hull and its region
            properties.
    """
    assert processed_stack.ndim == 3
    data_is_2D = processed_stack.shape[0] == 1
    if data_is_2D:
        processed_stack = processed_stack.reshape(processed_stack.shape[1:])
        hull = morphology.convex_hull_image(processed_stack)
    else:
        if no_copy:
            filtered_regions = get_filtered_stack_no_copy(
                shape_out=processed_stack.shape,
                regions=props,
                threshold=trac_nb_thresh,
                plane=plane,
            )
            hull = convex_hull_from_coordinates(
                coords=[region.centroid for region in filtered_regions],
                output_shape=processed_stack.shape,
            )
        else:
            filtered_hull_slices = get_filtered_stack(
                image_stack=processed_stack,
                regions=props,
                threshold=trac_nb_thresh,
                plane=plane,
            )
            hull = morphology.convex_hull_image(filtered_hull_slices)

    labeled_hull = measure.label(hull).astype(np.uint8)
    # uint8 is enough here because only 1 region is present
    if data_is_2D:
        hull_props = measure.regionprops(
            labeled_hull,
            spacing=(
                metadata["pixel_microns"],  # y
                metadata["pixel_microns"],  # x
            ),
        )
        hull_props_table = measure.regionprops_table(
            label_image=labeled_hull,
            properties=(
                "label",
                "bbox",
                "centroid",
                "area",
                "num_pixels",
                "slice",
                "equivalent_diameter_area",
                "solidity",
            ),
            spacing=(
                metadata["pixel_microns"],  # y
                metadata["pixel_microns"],  # x
            ),
        )
    else:
        hull_props = measure.regionprops(
            labeled_hull,
            spacing=(
                np.mean(np.diff(np.array(metadata["z_coordinates"]))),  # z
                metadata["pixel_microns"],  # y
                metadata["pixel_microns"],  # x
            ),
        )
        hull_props_table = measure.regionprops_table(
            label_image=labeled_hull,
            properties=(
                "label",
                "bbox",
                "centroid",
                "area",
                "num_pixels",
                "slice",
                "equivalent_diameter_area",
                "solidity",
            ),
            spacing=(
                np.mean(np.diff(np.array(metadata["z_coordinates"]))),  # z
                metadata["pixel_microns"],  # y
                metadata["pixel_microns"],  # x
            ),
        )
    return hull, hull_props, hull_props_table


def count_crossed_bboxes(
    plane: Plane,
    regions: list[Region],
    shape: tuple,
) -> list:
    """Scans a volume in space and counts the number of regions crossed
    by every plane in a specified plane familly.

    Args:
        plane (Plane): The plane familly along which the regions
        will be counted.
        regions (list[Region]): A list of regions to be counted.
        See Region class for more information.
        shape (tuple): The shape of the space to be scanned.

    Returns:
        list: Contains as elements the numbers of regions crossed
        by every plane in the specified familly. Indices represent
        the indices on the axis along which the counting occurs.
    """
    span = shape[plane.normal_axis]
    bboxes_crossed_by_slices = [0] * span

    for i in range(span):
        for region in regions:
            if plane.crosses(region, index=i):
                bboxes_crossed_by_slices[i] += 1

    return bboxes_crossed_by_slices


def find_unconsecutives(
    lst: list[int],
    sort: bool = True,
) -> list[tuple[int, int]]:
    """Scans a list of intengers for elements that are not consecutive
    and return their indices in a list of tuples.

    Args:
        lst (list[int]): A list of intengers.
        sort (bool, optional): If True, the list is sorted prior
        to finding the unconsecutivities. Defaults to True.

    Returns:
        list: A list of tuples containing the indices of the elements
        bordering the unconsecutivities.
    """
    if sort:
        lst_copy = sorted(lst)
    else:
        lst_copy = lst.copy()

    unconsecutive_indices = []

    for i in range(1, len(lst_copy)):
        if lst_copy[i] != lst_copy[i - 1] + 1:
            unconsecutive_indices.append((i - 1, i))

    return unconsecutive_indices


def fill_unconsecutive(
    lst: list[int],
    unconsecutive_indices: list[tuple[int, int]],
):
    """Fills in the gap of a list given list of its unconsecutivies.

    Args:
        lst (list[int]): The list to be filled in.
        unconsecutive_indices (list[tuple[int, int]]): The list of its
        unceonsecutivities in the form of tuples containing the indices
        of the elements bordering the unconsecutivities.

    Returns:
        list: The filled in list with only consecutive elements.
    """
    lst_copy = lst.copy()

    for start, stop in unconsecutive_indices:
        lst_copy.extend(list(range(lst[start] + 1, lst[stop])))

    lst_copy.sort()

    return lst_copy


# @profile
def get_hull_slices(image_stack: np.ndarray, plane: Plane) -> np.ndarray:
    """Scans a stack of binary images and computes the convex hull
    of each binary image and returns it as a binary image stack.

    Args:
        image_stack (np.ndarray): A stack a binary images.
        plane (Plane): The plane family giving the axis along which
        the scan is performed.

    Returns:
        np.ndarray: A stack of convex hull images computed from each
        slice in image_stack.
    """

    hull_stack = np.zeros(image_stack.shape)
    for i in range(image_stack.shape[plane.normal_axis]):
        hull_stack[plane.plane_selection_tuple(i)] = morphology.convex_hull_image(
            image_stack[plane.plane_selection_tuple(i)]
        )
    return hull_stack


# @profile
def get_filtered_stack(
    image_stack: np.ndarray,
    regions: list[Region],
    threshold: int,
    plane: Plane,
) -> np.ndarray:
    """Filters out the slices in an image stack which are below
    a threshold number of regions in them while keeping consecutivity
    of the slices.

    Args:
        image_stack (np.ndarray): A binary image stack, typically
        containning patches of True values (eg. tracers image stack).
        regions (list[Region]): A list of regions corresponding to
        the patches in the image stack, following the Region protocol.
        Typically the result of skimage.measure.regionprops() call.
        threshold (int): A threshold value below which a slice
        is filtered out.
        plane (Plane): A plane family in which the regions are counted.

    Returns:
        np.ndarray: A copy of the inital image stack with filtered out
        slices set to all False.
    """
    # count the number of bboxes crossed by each plane
    bboxes_crossed_by_slices = count_crossed_bboxes(
        plane=plane,
        regions=regions,
        shape=image_stack.shape,
    )

    # remove the planes with too little amount of tracers
    filtered_slices_indices = [
        i
        for i, elt in enumerate(
            bboxes_crossed_by_slices,
        )
        if elt > threshold
    ]

    # check if the remaining slices are still consecutive
    unconsecutive_indices = find_unconsecutives(
        lst=filtered_slices_indices,
        sort=False,
    )
    filled_slices_indices = fill_unconsecutive(
        lst=filtered_slices_indices,
        unconsecutive_indices=unconsecutive_indices,
    )

    # compute convex hull with remaining planes
    # filtered_hull_slices = get_hull_slices(
    #     image_stack=image_stack[plane.slice(filled_slices_indices)],
    #     plane=plane,
    # )

    # or just get the slices without computing the convex hull on each one
    # filtered_stack = image_stack[plane.slice(filled_slices_indices)]

    # making sure the input and output stack are the same size
    filtered_stack = image_stack.copy()
    filled_slices_indices = {
        plane.plane_selection_tuple(i) for i in filled_slices_indices
    }
    excluded_slice_indices = {
        plane.plane_selection_tuple(i)
        for i in range(
            image_stack.shape[plane.normal_axis],
        )
    }
    excluded_slice_indices.difference_update(filled_slices_indices)

    for i in excluded_slice_indices:
        filtered_stack[i] = 0

    # return filtered_hull_slices
    return filtered_stack


def get_filtered_stack_no_copy(
    shape_out: tuple,
    regions: list[Region],
    threshold: int,
    plane: Plane,
) -> np.ndarray:

    regions_cpy = regions.copy()
    # count the number of bboxes crossed by each plane
    bboxes_crossed_by_slices = count_crossed_bboxes(
        plane=plane,
        regions=regions,
        shape=shape_out,
    )

    # remove the planes with too little amount of tracers
    filtered_slices_indices = [
        i
        for i, elt in enumerate(
            bboxes_crossed_by_slices,
        )
        if elt > threshold
    ]

    # check if the remaining slices are still consecutive
    unconsecutive_indices = find_unconsecutives(
        lst=filtered_slices_indices,
        sort=False,
    )
    filled_slices_indices = fill_unconsecutive(
        lst=filtered_slices_indices,
        unconsecutive_indices=unconsecutive_indices,
    )

    # compute convex hull with remaining planes
    # filtered_hull_slices = get_hull_slices(
    #     image_stack=image_stack[plane.slice(filled_slices_indices)],
    #     plane=plane,
    # )

    # or just get the slices without computing the convex hull on each one
    # filtered_stack = image_stack[plane.slice(filled_slices_indices)]

    # making sure the input and output stack are the same size
    filled_slices_indices = {
        plane.plane_selection_tuple(i) for i in filled_slices_indices
    }
    excluded_slice_indices = {
        plane.plane_selection_tuple(i)
        for i in range(
            shape_out[plane.normal_axis],
        )
    }
    excluded_slice_indices.difference_update(filled_slices_indices)

    for region in regions_cpy:
        z = int(region.centroid[0])
        if z in excluded_slice_indices:
            regions_cpy.remove(region)

    # return filtered_regionprops
    return regions_cpy


def convex_hull_from_coordinates(coords, output_shape):
    """
    Generate a binary mask of the convex hull from a list of foreground voxel coordinates.

    Parameters:
    ----------
    coords : ndarray of shape (n_points, 3)
        Array of coordinates of the foreground voxels (points).
    output_shape : tuple of ints
        Shape of the output binary mask (e.g., the dimensions of the original image).

    Returns:
    -------
    convex_hull_mask : ndarray of shape output_shape
        A binary mask where the convex hull of the input points is set to True.

    Raises:
    -------
    ValueError: If any of the coordinates are out of bounds of the output shape.
    """
    coords = np.asarray(coords)

    # Validate coordinates
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords should be a 2D array with shape (n_points, 3).")
    if np.any(coords < 0) or np.any(coords >= np.array(output_shape)):
        raise ValueError(
            "Some coordinates are out of bounds for the given output_shape."
        )

    # Compute convex hull
    hull = ConvexHull(coords)

    # Create an empty binary mask
    mask = np.zeros(output_shape, dtype=bool)

    # Rasterize the convex hull by filling it in
    vertices = coords[hull.vertices]  # Points on the hull boundary
    grid_x, grid_y, grid_z = np.indices(output_shape)

    # Use inequalities of the hull's facets to identify points inside the hull
    for simplex in hull.simplices:
        # Extract vertices of the simplex (a triangular face in 3D)
        tri = vertices[simplex]

        # Compute the plane equation (Ax + By + Cz + D = 0)
        # Normal vector (A, B, C) and constant D
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        D = -np.dot(normal, tri[0])

        # Check which points are below the plane (inside the convex hull)
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))
        mask |= np.dot(grid_points, normal) + D <= 0

    # Fill holes to complete the convex hull
    convex_hull_mask = binary_fill_holes(mask)
    return convex_hull_mask
