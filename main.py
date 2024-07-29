from tkinter.filedialog import askopenfilename

import numpy as np
from matplotlib import pyplot as plt

from load_image_stack import load_image_stack
from process_stack import process_stack
from multi_slice_viewer import multi_slice_viewer
from particle_distributions import particle_distributions
from utils import z_resolution


# def main() -> None:
path = askopenfilename()

image_stack, metadata = load_image_stack(path)

results = process_stack(image_stack, metadata, 1, full_bbox=False)

multi_slice_viewer(
    # volume=results["binary"],
    volume=image_stack,
    bboxes=results["bboxes3d"],
    bbox_alpha=0.5,
)

plt.show()

total_volume = (
    abs(metadata["z_coordinates"][-1] - metadata["z_coordinates"][0])
    * metadata["width"]
    * metadata["height"]
    * metadata["pixel_microns"] ** 2
)

num_particles = len(results["props"])

particle_concentration = num_particles / total_volume

print(f"Total volume analysed: {total_volume:.4e} µm^3")
print(f"Number of particles detected: {num_particles:.4e}")
print(f"Particle concentration: {particle_concentration:.4e} particles/µm^3")

x, y, z = particle_distributions(
    results,
    metadata,
    bins_xy=20,
    bins_z=20,
    bins_pixels=100,
    bins_area=100,
    bins_diameter=100,
    boxplot_config={"showfliers": False},
    # verbose=True,
)

median_diameter = np.mean(np.array([np.median(x), np.median(y)]))
median_num_pixels = np.median(
    np.array(
        [prop.num_pixels for prop in results["props"]],
    )
)

results = process_stack(
    image_stack=image_stack,
    metadata=metadata,
    particle_diameter=1,
    full_bbox=False,
    spacing=(
        z_resolution(
            diameter=median_diameter,
            num_pixels=median_num_pixels,
            x_resolution=metadata["pixel_microns"],
            y_resolution=metadata["pixel_microns"],
        ),
        metadata["pixel_microns"],
        metadata["pixel_microns"],
    ),
)

particle_volumes = np.array(
    [prop.area for prop in results["spacing_corrected_props"]],
)
num_particles_in_volume = np.sum(particle_volumes) / np.median(particle_volumes)
particle_concentration_in_volume = num_particles_in_volume / total_volume

print(
    f"Number of particles (computed in volume): {num_particles_in_volume:.4e}",
)
print(
    "Particle concentration (computed in volume):"
    f"{particle_concentration_in_volume:.4e} particles/µm^3"
)

plt.show()

# if __name__ == "__main__":
#     main()
