from tkinter.filedialog import askopenfilename

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.special import j1
from scipy.stats import zscore
from scipy.ndimage import center_of_mass
from skimage import io, feature, filters, restoration, measure

from load_image_stack import load_image_stack

# * Load particles image stack
path = askopenfilename()

image_stack, metadata = load_image_stack(path)

# Preprocess the image (optional)
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

# Extract PSF for each detected particle
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

# Align PSFs to a common center
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

# Removing cropped PSFs
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

# Compute the final mean PSF without outliers
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
)  # Radius of the ideal sphere in pixels

# Generate the theoretical PSF
theoretical_psf = generate_theoretical_psf(psf_shape, sphere_radius)

# Deconvolve observed PSF with theoretical PSF
num_iter = 30
restored_psf = restoration.richardson_lucy(
    mean_psf_smoothed, theoretical_psf, num_iter=num_iter
)
deconvolution_test = restoration.richardson_lucy(
    mean_psf_smoothed, restored_psf, num_iter=num_iter
)

# ? Fit restored PSF using gaussian or Airy model

# * Save the result
# - as an array if no model is used
# - only the parameters resulting from the fit otherwise

# Save the final PSF
np.save("restored_psf.npy", restored_psf)

# * Display the results
# Visualization of the restored PSF
# In the xy pane
fig_xy, ax_xy = plt.subplots(2, 3)

ax_xy[0, 0].set_xlabel("Averaged PSF with ouliers (Central Slice)")
img = ax_xy[0, 0].imshow(initial_mean_psf[initial_mean_psf.shape[0] // 2], cmap="gray")

ax_xy[0, 1].set_title("xy plane")
ax_xy[0, 1].set_xlabel("Averaged PSF without ouliers (Central Slice)")
ax_xy[0, 1].imshow(final_mean_psf[final_mean_psf.shape[0] // 2], cmap="gray")

ax_xy[0, 2].set_xlabel("Smoothed PSF (Central Slice)")
ax_xy[0, 2].imshow(mean_psf_smoothed[mean_psf_smoothed.shape[0] // 2], cmap="gray")

ax_xy[1, 0].set_xlabel("Theorical spherical PSF (Central Slice)")
ax_xy[1, 0].imshow(theoretical_psf[theoretical_psf.shape[0] // 2], cmap="gray")

ax_xy[1, 1].set_xlabel(f"Restored PSF, num_iter={num_iter} (Central Slice)")
ax_xy[1, 1].imshow(restored_psf[restored_psf.shape[0] // 2], cmap="gray")

ax_xy[1, 2].set_xlabel(f"Deconvolution test, num_iter={num_iter} (Central Slice)")
ax_xy[1, 2].imshow(deconvolution_test[deconvolution_test.shape[0] // 2], cmap="gray")

plt.colorbar(img, cax=None, ax=ax_xy)

# In the xz pane
fig_xz, ax_xz = plt.subplots(2, 3)

ax_xz[0, 0].set_xlabel("Averaged PSF with ouliers (Central Slice)")
img = ax_xz[0, 0].imshow(
    initial_mean_psf[:, initial_mean_psf.shape[1] // 2, :], cmap="gray"
)

ax_xz[0, 1].set_title("xz plane")
ax_xz[0, 1].set_xlabel("Averaged PSF without ouliers (Central Slice)")
ax_xz[0, 1].imshow(final_mean_psf[:, final_mean_psf.shape[1] // 2, :], cmap="gray")

ax_xz[0, 2].set_xlabel("Smoothed PSF (Central Slice)")
ax_xz[0, 2].imshow(
    mean_psf_smoothed[:, mean_psf_smoothed.shape[1] // 2, :], cmap="gray"
)

ax_xz[1, 0].set_xlabel("Theorical spherical PSF (Central Slice)")
ax_xz[1, 0].imshow(theoretical_psf[:, theoretical_psf.shape[1] // 2, :], cmap="gray")

ax_xz[1, 1].set_xlabel(f"Restored PSF, num_iter={num_iter} (Central Slice)")
ax_xz[1, 1].imshow(restored_psf[:, restored_psf.shape[1] // 2, :], cmap="gray")

ax_xz[1, 2].set_xlabel(f"Deconvolution test, num_iter={num_iter} (Central Slice)")
ax_xz[1, 2].imshow(
    deconvolution_test[:, deconvolution_test.shape[1] // 2, :], cmap="gray"
)

plt.colorbar(img, cax=None, ax=ax_xz)

# In the yz pane
fig_yz, ax_yz = plt.subplots(2, 3)

ax_yz[0, 0].set_xlabel("Averaged PSF with ouliers (Central Slice)")
img = ax_yz[0, 0].imshow(
    initial_mean_psf[..., initial_mean_psf.shape[2] // 2], cmap="gray"
)

ax_yz[0, 1].set_title("yz plane")
ax_yz[0, 1].set_xlabel("Averaged PSF without ouliers (Central Slice)")
ax_yz[0, 1].imshow(final_mean_psf[..., final_mean_psf.shape[2] // 2], cmap="gray")

ax_yz[0, 2].set_xlabel("Smoothed PSF (Central Slice)")
ax_yz[0, 2].imshow(mean_psf_smoothed[..., mean_psf_smoothed.shape[2] // 2], cmap="gray")

ax_yz[1, 0].set_xlabel("Theorical spherical PSF (Central Slice)")
ax_yz[1, 0].imshow(theoretical_psf[..., theoretical_psf.shape[2] // 2], cmap="gray")

ax_yz[1, 1].set_xlabel(f"Restored PSF, num_iter={num_iter} (Central Slice)")
ax_yz[1, 1].imshow(restored_psf[..., restored_psf.shape[2] // 2], cmap="gray")

ax_yz[1, 2].set_xlabel(f"Deconvolution test, num_iter={num_iter} (Central Slice)")
ax_yz[1, 2].imshow(
    deconvolution_test[..., deconvolution_test.shape[2] // 2], cmap="gray"
)

plt.colorbar(img, cax=None, ax=ax_yz)

plt.show()
