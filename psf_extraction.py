from tkinter.filedialog import askopenfilename
from pathlib import Path
from enum import StrEnum, auto
from functools import partial

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.special import j1
from scipy.stats import zscore
from scipy.ndimage import center_of_mass
from skimage import io, feature, filters, restoration, measure

from files_io import load_image_stack
from plane import Plane

"""
Pour l'instant la déconvolution donne des cubes. Peut-être que c'est pas
grave, même pour le comptage en volume.
- voir ce que ça donne un stack déconvolué en entier (séparation des 
agrégats ?, problèmes ?)
- comparer comptage en volume et en nombre à tester sur un stack
déconvolué et non déconvolué
- comparer le calcul du volume de capsule avec et sans déconvolution
"""


def plot_psf_results(
    initial_mean_psf: np.ndarray,
    final_mean_psf: np.ndarray,
    mean_psf_smoothed: np.ndarray,
    theoretical_psf: np.ndarray,
    restored_psf: np.ndarray,
    deconvolution_test: np.ndarray,
    plane: Plane | str,
    num_iter_restoration: int,
    num_iter_test: int,
) -> tuple:
    plane = Plane(plane)
    fig, ax = plt.subplots(2, 3)

    ax[0, 0].set_xlabel("Averaged PSF\nwith ouliers")
    # fmt: off
    img = ax[0, 0].imshow(
        initial_mean_psf[
            plane.middle_slice(initial_mean_psf.shape[plane.normal_axis])
        ],
        cmap="gray",
    )
    # fmt: on

    ax[0, 1].set_title(f"{plane} plane\n(central slices)")
    ax[0, 1].set_xlabel("Averaged PSF\nwithout ouliers")
    # fmt: off
    ax[0, 1].imshow(
        final_mean_psf[
            plane.middle_slice(final_mean_psf.shape[plane.normal_axis])
        ],
        cmap="gray",
    )
    # fmt: on

    ax[0, 2].set_xlabel("Smoothed PSF")
    # fmt: off
    ax[0, 2].imshow(
        mean_psf_smoothed[
            plane.middle_slice(mean_psf_smoothed.shape[plane.normal_axis])
        ],
        cmap="gray",
    )
    # fmt: on

    ax[1, 0].set_xlabel("Theorical spherical PSF")
    # fmt: off
    ax[1, 0].imshow(
        theoretical_psf[
            plane.middle_slice(theoretical_psf.shape[plane.normal_axis])
        ],
        cmap="gray",
    )
    # fmt: on

    ax[1, 1].set_xlabel(f"Restored PSF,\nnum_iter={num_iter_restoration}")
    # fmt: off
    ax[1, 1].imshow(
        restored_psf[
            plane.middle_slice(restored_psf.shape[plane.normal_axis])
        ],
        cmap="gray",
    )
    # fmt: on

    ax[1, 2].set_xlabel(f"Deconvolution test,\nnum_iter={num_iter_test}")
    # fmt: off
    ax[1, 2].imshow(
        deconvolution_test[
            plane.middle_slice(deconvolution_test.shape[plane.normal_axis])
        ],
        cmap="gray",
    )
    # fmt: on
    plt.colorbar(img, cax=None, ax=ax)

    fig.set_layout_engine(layout="constrained")

    return fig, ax


def main(path: str | Path):
    # ? on peut jouer sur:
    # - le filtre gaussien en pre-process et post-process
    # - les dimensions de la zone extraite autour des traceurs
    # - le z-score pour retirer les outliers
    # - la forme du PSF théorique
    # ! - le nombre d'itérations de l'algo de déconvolution pour extraire le PSF
    # ! - le nombre d'itérations de l'algo de déconvolution pour tester le PSF

    # * Load particles image stack
    image_stack, metadata = load_image_stack(str(path))
    # todo : take the config file into account

    # * Preprocess the image (optional)
    # Example: Gaussian filter to reduce noise
    image_3d_filtered = filters.gaussian(image_stack, sigma=1)

    # * Extracting PSFs
    # *---------------------------------------------------------------------
    # Detect particles using a local maxima detector
    # Adjust the parameters to suit your data
    threshold = filters.threshold_otsu(image_3d_filtered)
    particles = feature.peak_local_max(
        image_3d_filtered, threshold_abs=threshold, min_distance=5
    )
    print(f"Particles detected initially: {len(particles)}")

    # Define the size of the PSF region to extract around each particle
    psf_shape = np.array([61, 21, 21])  # z, y, x
    half_size = psf_shape // 2

    # List to store individual PSFs
    psfs = []

    # * Extract PSF for each detected particle
    for coord in particles:
        z, y, x = coord
        # Define the region around the particle
        z_start = max(z - half_size[0], 0)
        z_end = min(z + half_size[0] + 1, image_stack.shape[0])
        y_start = max(y - half_size[1], 0)
        y_end = min(y + half_size[1] + 1, image_stack.shape[1])
        x_start = max(x - half_size[2], 0)
        x_end = min(x + half_size[2] + 1, image_stack.shape[2])

        # Extract the PSF region
        psf_region = image_stack[z_start:z_end, y_start:y_end, x_start:x_end]

        # Normalize and store the PSF
        psf_region = psf_region / np.sum(psf_region)
        psfs.append(psf_region)

    # * Align PSFs to a common center
    # Assume the particle's center is the peak intensity point
    aligned_psfs = []
    for psf in psfs:
        # Find the center of mass
        shift = half_size - center_of_mass(psf)
        shifted_psf = np.roll(
            np.roll(
                np.roll(
                    psf,
                    int(shift[0]),
                    axis=0,
                ),
                int(shift[1]),
                axis=1,
            ),
            int(shift[2]),
            axis=2,
        )
        aligned_psfs.append(shifted_psf)

    # * Removing cropped PSFs
    # fmt: off
    aligned_psfs = [
        psf for psf in aligned_psfs if (psf.shape == psf_shape).all()
    ]
    print(f"Number of bordering particles: {len(psfs) - len(aligned_psfs)}")
    # fmt: on
    # *---------------------------------------------------------------------

    # * Removing outliers
    # Compute the initial mean PSF as the reference
    initial_mean_psf = np.mean(aligned_psfs, axis=0)

    # Calculate similarity metric (Sum of Squared Differences) for each PSF
    ssd_values = []
    for psf in aligned_psfs:
        ssd = np.sum((psf - initial_mean_psf) ** 2)
        ssd_values.append(ssd)

    # Convert SSD values to z-scores to identify outliers
    z_scores = zscore(ssd_values)

    # Set a threshold for outlier detection (e.g., |z| > 2)
    outlier_threshold = 1
    outliers = np.abs(z_scores) > outlier_threshold

    # Remove outliers
    filtered_psfs = [psf for i, psf in enumerate(aligned_psfs) if not outliers[i]]
    print(f"Number of outliers: {np.sum(outliers)}")

    # * Compute the final mean PSF without outliers
    final_mean_psf = np.mean(filtered_psfs, axis=0)

    # ? Optionally refine the PSF (e.g., smooth)
    # Smooth the average PSF
    mean_psf_smoothed = filters.gaussian(final_mean_psf, sigma=1)

    # * Restore PSF by deconvolution
    # Generate theoretical PSF for an ideal sphere
    def generate_theoretical_psf(shape, sphere_radius):
        """Generate a 3D PSF for an ideal fluorescent sphere."""
        x_center = shape[2] // 2
        y_center = shape[1] // 2
        z_center = shape[0] // 2
        x = np.arange(shape[2])
        y = np.arange(shape[1])
        z = np.arange(shape[0])
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
        # fmt: off
        distance = np.sqrt(
            (X - x_center) ** 2
            + (Y - y_center) ** 2
            + (Z - z_center) ** 2
        )
        # fmt: on
        sphere_psf = (distance <= sphere_radius).astype(float)
        sphere_psf /= np.sum(sphere_psf)  # Normalize the theoretical PSF
        return sphere_psf

    # Parameters for the theoretical PSF
    psf_shape = mean_psf_smoothed.shape  # Assuming the PSF is a cube
    sphere_radius = round(
        1 / metadata["pixel_microns"]
    )  # Radius of the ideal 1 micron sphere in pixels

    # Generate the theoretical PSF
    theoretical_psf = generate_theoretical_psf(psf_shape, sphere_radius)

    # Deconvolve observed PSF with theoretical PSF
    num_iter_restoration = 5
    num_iter_test = 100
    restored_psf = restoration.richardson_lucy(
        mean_psf_smoothed, theoretical_psf, num_iter=num_iter_restoration
    )
    deconvolution_test = restoration.richardson_lucy(
        mean_psf_smoothed, restored_psf, num_iter=num_iter_test
    )

    # ? Fit restored PSF using gaussian or Airy model

    # * Save the result
    # - as an array if no model is used
    # - only the parameters resulting from the fit otherwise

    # Save the final PSF
    np.save("restored_psf.npy", restored_psf)

    # * Display the results
    # Visualization of the restored PSF

    plot_results = partial(
        plot_psf_results,
        initial_mean_psf,
        final_mean_psf,
        mean_psf_smoothed,
        theoretical_psf,
        restored_psf,
        deconvolution_test,
        num_iter_restoration=num_iter_restoration,
        num_iter_test=num_iter_test,
    )

    print(f"num_iter_restoration = {num_iter_restoration}")
    print(f"num_iter_test = {num_iter_test}")

    # In the xy pane
    fig_xy, ax_xy = plot_results(plane=Plane.XY)

    # In the xz pane
    fig_xz, ax_xz = plot_results(plane=Plane.XZ)

    # In the yz pane
    fig_yz, ax_yz = plot_results(plane=Plane.YZ)

    plt.show()


if __name__ == "__main__":
    path = Path(
        r"C:\Users\aguirrep\Documents\microscopie\fluo_tests\comptage"
        r"\data\calib-caps\1.5pc-20-caps1.nd2"
    )
    # path = askopenfilename()
    main(path)
