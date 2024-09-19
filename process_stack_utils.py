import numpy as np
from skimage import filters, measure, morphology, draw

from plane import Plane, Region


def threshold_stack(
    image_stack: np.ndarray,
    thresh: int | None = None,
) -> tuple[int, np.ndarray]:

    if not thresh:
        thresh = filters.threshold_otsu(image_stack)
    processed_stack = image_stack > thresh
    return thresh, processed_stack


def morphological_opening(
    processed_stack: np.ndarray,
    metadata: dict,
    particle_diameter_um: float,
) -> np.ndarray:
    structuring_element = morphology.ball(  # radius in pixels:
        max(1, particle_diameter_um / metadata["pixel_microns"] / 2)
    )

    processed_stack = morphology.binary_opening(
        image=processed_stack,
        footprint=structuring_element,
    )
    return processed_stack


def measure_particles(processed_stack: np.ndarray, metadata: dict) -> None:
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
    return labels, props, scaled_props


def create_bboxes_stack(
    stack_shape: tuple[int, int, int],
    props: list[Region],
    full_bboxes: bool = False,
) -> None:
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


def get_spacing_correction(
    labels: np.ndarray,
    spacing: tuple[int, int, int],
) -> None:
    spacing_corrected_props = measure.regionprops(
        label_image=labels,
        spacing=spacing,
    )
    return spacing_corrected_props


def make_centroids_stack(props: Region, shape: tuple[int, int, int]):

    centroids_coords = [prop.centroid for prop in props]
    centroids_stack = np.zeros(shape=shape, dtype=bool)
    for coords in centroids_coords:
        try:
            rounded_coords = tuple([int(coord.round()) for coord in coords])
            centroids_stack[rounded_coords] = True
        except IndexError:
            pass

    return centroids_stack


def process_capsule_hull(
    processed_stack: np.ndarray,
    props: list[Region],
    metadata: dict,
    plane: Plane = Plane.XY,
) -> None:
    filtered_hull_slices = get_filtered_stack(
        image_stack=processed_stack,
        regions=props,
        threshold=20,
        plane=plane,
    )
    hull = morphology.convex_hull_image(filtered_hull_slices)
    labeled_hull = measure.label(hull)
    hull_props = measure.regionprops(
        labeled_hull,
        spacing=(
            np.mean(np.diff(np.array(metadata["z_coordinates"]))),  # z
            metadata["pixel_microns"],  # y
            metadata["pixel_microns"],  # x
        ),
    )
    return hull, hull_props


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


def get_filtered_stack(
    image_stack: np.ndarray,
    regions: list[Region],
    threshold: int,
    plane: Plane,
) -> np.ndarray:
    """Filters out the slices  in an image stack which a below
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
    filtered_slices_inidces = [
        i
        for i, elt in enumerate(
            bboxes_crossed_by_slices,
        )
        if elt > threshold
    ]

    # check if the remaining slices are still consecutive
    unconsecutive_indices = find_unconsecutives(
        lst=filtered_slices_inidces,
        sort=False,
    )
    filled_slices_indices = fill_unconsecutive(
        lst=filtered_slices_inidces,
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
