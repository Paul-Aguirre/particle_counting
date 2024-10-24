import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from datetime import datetime


def generate_image_stack(
    width,  # Image width in pixels
    height,  # Image height in pixels
    pixel_size,  # Pixel size in micrometers
    num_slices,  # Number of z-slices (images along the z-axis)
    z_step,  # Step size between z-slices in micrometers
    microsphere_size,  # Diameter of microspheres in micrometers
    microsphere_count,  # Number of microspheres in the volume
    blur_sigma_x,  # Spread/blur in x-direction (Gaussian sigma in pixels)
    blur_sigma_y,  # Spread/blur in y-direction (Gaussian sigma in pixels)
    blur_sigma_z,  # Spread/elongation in z-direction (Gaussian sigma in pixels)
    grayscale=True,  # Whether to generate grayscale (True) or binary (False) images
):
    # Initialize an empty 3D stack (using float32 to save memory)
    img_stack = np.zeros((num_slices, height, width), dtype=np.float32)

    # Determine the radius of the microspheres in pixels
    sphere_radius_pix = microsphere_size / 2 / pixel_size

    # Randomly place microspheres within the 3D volume
    microsphere_positions = np.random.uniform(
        [sphere_radius_pix, sphere_radius_pix, sphere_radius_pix],
        [
            width - sphere_radius_pix,
            height - sphere_radius_pix,
            num_slices - 1 - sphere_radius_pix,
        ],
        size=(microsphere_count, 3),
    )

    # Add microspheres to the 3D image stack
    for pos in microsphere_positions:
        x, y, z = pos

        # Generate a 3D Gaussian (blurred sphere) or raw unblurred sphere
        for slice_idx in range(num_slices):
            # Calculate the distance of the slice from the center of the microsphere in z
            z_dist = (slice_idx - z) * z_step / pixel_size

            # Create a circular mask for each z-slice based on microsphere size and position
            yy, xx = np.meshgrid(np.arange(height), np.arange(width))
            dist_from_center = np.sqrt((xx - x) ** 2 + (yy - y) ** 2 + z_dist**2)

            # For grayscale mode, generate fluorescence intensity
            if grayscale:
                sphere_intensity = np.exp(
                    -((dist_from_center / sphere_radius_pix) ** 2)
                )
                img_stack[slice_idx] += sphere_intensity
            else:
                # For binary mode, mark the pixels within the radius of the microsphere as True
                img_stack[slice_idx] |= dist_from_center <= sphere_radius_pix

    # For grayscale images, normalize intensity to simulate weak signals with 16-bit depth
    if grayscale:
        # Normalize and scale the intensities to peak at ~4000
        img_stack = img_stack / img_stack.max() * 4000

        # Apply anisotropic Gaussian blur to simulate spot elongation if needed
        if blur_sigma_x > 0 or blur_sigma_y > 0 or blur_sigma_z > 0:
            # Gaussian filter requires sigma per axis, so we create a tuple
            sigma = (blur_sigma_z, blur_sigma_x, blur_sigma_y)
            img_stack = gaussian_filter(img_stack, sigma=sigma)

        # Normalize intensities again to ensure proper 16-bit scaling after blurring
        img_stack = img_stack / img_stack.max() * 4000

        # Convert to 16-bit integer values
        img_stack = img_stack.astype(np.uint16)

    # Prepare metadata dictionary with all input parameters and extra metadata
    metadata = {
        # Input parameters
        "width": width,
        "height": height,
        "pixel_microns": pixel_size,
        "num_slices": num_slices,
        "z_step": z_step,
        "microsphere_size": microsphere_size,
        "microsphere_count": microsphere_count,
        "blur_sigma_x": blur_sigma_x,
        "blur_sigma_y": blur_sigma_y,
        "blur_sigma_z": blur_sigma_z,
        "grayscale": grayscale,
        # Additional metadata
        "z_coordinates": [i * z_step for i in range(num_slices)],  # z-positions
        "z_levels": range(num_slices),  # z-slice indices (0 to num_slices-1)
        "date": datetime.now(),  # Timestamp when the image stack was generated
    }

    return img_stack, metadata


# Example usage:
width = 128  # Image width in pixels
height = 128  # Image height in pixels
pixel_size = 0.353  # Pixel size in micrometers
num_slices = 40  # Number of z-slices
z_step = 0.6  # Step size along the z-axis in micrometers
microsphere_size = 1  # Diameter of microspheres in micrometers
microsphere_count = 40  # Number of microspheres in the volume
blur_sigma_x = 1.0  # Spread in x-direction (Gaussian sigma in pixels)
blur_sigma_y = 1.0  # Spread in y-direction (Gaussian sigma in pixels)
blur_sigma_z = 3.0  # Elongation in z-direction (Gaussian sigma in pixels)

# Generate the image stack (grayscale) and metadata
img_stack, metadata = generate_image_stack(
    width=width,
    height=height,
    pixel_size=pixel_size,
    num_slices=num_slices,
    z_step=z_step,
    microsphere_size=microsphere_size,
    microsphere_count=microsphere_count,
    blur_sigma_x=blur_sigma_x,
    blur_sigma_y=blur_sigma_y,
    blur_sigma_z=blur_sigma_z,
    grayscale=True,  # Change to False for binary images
)

# Visualize some slices from the generated image stack
# fig, axes = plt.subplots(1, 5, figsize=(15, 3))
# for i, ax in enumerate(axes):
#     slice_idx = i * num_slices // 5
#     ax.imshow(img_stack[slice_idx], cmap="gray", vmin=0, vmax=65535)
#     ax.set_title(f"Z-slice {slice_idx}")
#     ax.axis("off")
# plt.tight_layout()
# plt.show()

# Print metadata
print("Metadata:")
for key, value in metadata.items():
    print(f"{key}: {value}")
