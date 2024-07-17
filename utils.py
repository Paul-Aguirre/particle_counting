def depth_of_field(
    *,
    numerical_aperture: float,
    magnification: float,
    smallest_dist: float,
    refraction_index: float = 1,  # for air
    wavelength: float = 550e-9,  # 550 nm in meters
) -> float:
    """Computes the approximate depth of field for a given set of
    arguments depending on the objective used. Formula is presented at
    https://www.microscopyu.com/tutorials/depthoffield

    Args:
        numerical_aperture (float): numerical aperture of the objective
        magnification (float): magification of the objective,
        eg. 10 for a 10x objective.
        smallest_dist (float): smallest distance that can be resolved
        by the detector that is placed in the image plane of the
        microscope objective.
        wavelength (float, optionnal): wavelength of the incident ligth in
        refraction_index (float, optionnal): refraction index of the media between
        the cover slip and the objective.
        1 for air, 1.515 for imersion oil, 1.333 for water.

    Returns:
        float: approximate depth of field in meters.
    """
    d_tot = (
        wavelength * refraction_index / numerical_aperture**2
        + refraction_index * smallest_dist / magnification / numerical_aperture
    )
    return d_tot
