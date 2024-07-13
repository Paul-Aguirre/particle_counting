from tkinter.filedialog import askopenfilename

from matplotlib import pyplot as plt

from load_image_stack import load_image_stack
from process_stack import process_stack
from multi_slice_viewer import multi_slice_viewer


# def main() -> None:
path = askopenfilename()

image_stack, metadata = load_image_stack(path)

results = process_stack(image_stack, metadata, 1)

multi_slice_viewer(
    volume=results["binary_separated"],
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

print(f"Total volume analysed: {total_volume:.3f} µm^3")
print(f"Number of particles detected: {num_particles}")
print(f"Particle concentration: {particle_concentration:.7f} particles/µm^3")

# if __name__ == "__main__":
#     main()
