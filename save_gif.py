from tkinter.filedialog import askopenfilename, asksaveasfilename
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.colors as mcolors

from files_io import load_image_stack


def apply_logarithmic_norm(image_stack: np.ndarray):
    """Applies a logarithmic norm to every image of an image stack and
    returns it as a list of PIL Images.

    Args:
        image_stack (np.ndarray): An image stack.

    Returns:
        list[PIL.Images]: List of normalised images.
    """
    # Create an empty list to store the normalized images
    normalised_images: list = []

    # LogNorm for normalization
    log_norm = mcolors.LogNorm()

    for i in range(image_stack.shape[0]):
        # Convert to float for processing
        img_array: np.ndarray = image_stack[i].astype(float)

        # Apply the LogNorm normalization
        img_norm: np.ndarray = log_norm(img_array)

        # Scale back to 0-255 (or 8-bit range)
        img_norm = (img_norm * 255).astype(np.uint8)

        # Convert the NumPy array back to a PIL image in 'L' mode (grayscale)
        normalised_img = Image.fromarray(img_norm, mode="L")

        # Append the normalized image to the list
        normalised_images.append(normalised_img)

    return normalised_images


def save_as_gif(
    images: list[Image.Image],
    gif_path: str | Path,
    duration: int | list[int] = 100,  # Duration between frames in milliseconds
    loop: int = 0,  # Loop forever
):
    """Turns a list of PIL.Images.Images into a grayscale GIF file.

    Args:
        images (list[Image.Image]): list of PIL.Image.Image to be turned
            into a gif.
        gif_path (str | Path): A filename (string), os.PathLike object
            or file object.
        duration (int | list[int], optional): The display duration of
            each frame of the multiframe gif, in milliseconds.
            Pass a single integer for a constant duration, or a list
            or tuple to set the duration for each frame separately.
            Defaults to 100.
        loop (int, optionnal): Integer number of times the GIF should
            loop. 0 means that it will loop forever. If omitted or None,
            the image will not loop. Defaults to 0.
    """
    # Ensure all images are in 'L' mode (grayscale format for GIFs)
    images = [img.convert("L") for img in images]

    # Save the list of images as a GIF
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],  # Append the rest of the frames
        duration=duration,  # Duration between frames in milliseconds
        loop=loop,
    )


def main():
    """Prompts the user to choose an ND2 file to convert into a GIF and
    ask where to save it.
    """
    image_stack, _ = load_image_stack(askopenfilename())
    normalised_images = apply_logarithmic_norm(image_stack)
    save_as_gif(normalised_images, asksaveasfilename())


if __name__ == "__main__":
    main()
