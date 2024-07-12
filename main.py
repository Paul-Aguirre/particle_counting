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

# if __name__ == "__main__":
#     main()
