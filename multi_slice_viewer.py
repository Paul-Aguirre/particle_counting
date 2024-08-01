from tkinter.filedialog import askopenfilename

import numpy as np
from matplotlib import pyplot as plt

from files_inputs import load_image_stack


def multi_slice_viewer(
    volume: np.ndarray,
    bboxes: np.ndarray | None = None,
    bbox_alpha: float = 0.5,
) -> None:

    remove_keymap_conflicts({"j", "k", "h", "l"})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index], cmap="gray")

    if bboxes is not None:
        assert bboxes.shape == volume.shape

        ax.bboxes = bboxes

        red_foreground = np.zeros((*volume.shape[1:], 4), dtype=np.float64)
        red_foreground[..., 0] = 1
        red_foreground[..., 3] = bboxes[ax.index] * bbox_alpha
        ax.red_foreground = red_foreground
        ax.bbox_alpha = bbox_alpha
        ax.imshow(red_foreground)

    else:
        ax.bboxes = None

    fig.canvas.mpl_connect("key_press_event", process_key)
    fig.canvas.manager.set_window_title(f"Slice {ax.index}/{volume.shape[0]}")

    return fig, ax


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == "j":
        previous_slice(ax)
    elif event.key == "k":
        next_slice(ax)
    elif event.key == "h":
        previous_jump(ax)
    elif event.key == "l":
        next_jump(ax)
    fig.canvas.draw()
    # fmt: off
    (fig.canvas.manager
    .set_window_title(f"Slice {ax.index}/{ax.volume.shape[0]}"))
    # fmt: on


def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    if ax.bboxes is not None:
        ax.red_foreground[..., 3] = ax.bboxes[ax.index] * ax.bbox_alpha
        ax.images[1].set_array(ax.red_foreground)


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    if ax.bboxes is not None:
        ax.red_foreground[..., 3] = ax.bboxes[ax.index] * ax.bbox_alpha
        ax.images[1].set_array(ax.red_foreground)


def previous_jump(ax):
    volume = ax.volume
    ax.index = (ax.index - volume.shape[0] // 10) % volume.shape[
        0
    ]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    if ax.bboxes is not None:
        ax.red_foreground[..., 3] = ax.bboxes[ax.index] * ax.bbox_alpha
        ax.images[1].set_array(ax.red_foreground)


def next_jump(ax):
    volume = ax.volume
    ax.index = (ax.index + volume.shape[0] // 10) % volume.shape[
        0
    ]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    if ax.bboxes is not None:
        ax.red_foreground[..., 3] = ax.bboxes[ax.index] * ax.bbox_alpha
        ax.images[1].set_array(ax.red_foreground)


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith("keymap."):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


if __name__ == "__main__":
    image_stack, metadata = load_image_stack(path=askopenfilename())
    multi_slice_viewer(volume=image_stack)
    plt.show()
