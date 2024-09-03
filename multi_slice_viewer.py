from tkinter.filedialog import askopenfilename
from pathlib import Path
from functools import partial, partialmethod
from typing import Callable

import matplotlib.axes
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib

from files_inputs import load_image_stack

# volume = None


class MultiSliceViewer:
    def __init__(
        self,
        lognorm: bool = False,
    ) -> None:

        self.key_bindings = self._create_key_bindings()
        self._remove_keymap_conflicts(set(self.key_bindings.keys()))
        self.fig, self.ax = plt.subplots()
        self.lognorm = lognorm

        self.fig.canvas.mpl_connect(
            "key_press_event",
            partial(
                MultiSliceViewer._process_key,
                key_bindings=self.key_bindings,
            ),
        )

    def plot(
        self,
        volume: np.ndarray,
        bboxes: np.ndarray | None = None,
        bbox_alpha: float = 0.5,
        cmap: str = "gray",
    ):
        self.ax.volume = volume
        self.ax.index = volume.shape[0] // 2

        if self.lognorm:
            norm = colors.LogNorm(vmin=volume.min(), vmax=volume.max())
        else:
            norm = colors.Normalize()

        self.ax.imshow(volume[self.ax.index], norm=norm, cmap=cmap)

        if bboxes is not None:
            assert bboxes.shape == volume.shape

            self.ax.bboxes = bboxes

            red_foreground = np.zeros(
                (*volume.shape[1:], 4),
                dtype=np.float64,
            )
            red_foreground[..., 0] = 1
            red_foreground[..., 3] = bboxes[self.ax.index] * bbox_alpha
            self.ax.red_foreground = red_foreground
            self.ax.bbox_alpha = bbox_alpha
            self.ax.imshow(red_foreground)

        else:
            self.ax.bboxes = None

        self.fig.canvas.manager.set_window_title(
            f"Slice {self.ax.index}/{volume.shape[0]}"
        )

    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def _create_key_bindings() -> dict[str, Callable]:
        return {
            "j": MultiSliceViewer._previous_slice,
            "k": MultiSliceViewer._next_slice,
            "h": MultiSliceViewer._previous_jump,
            "l": MultiSliceViewer._next_jump,
            "c": MultiSliceViewer._crop_xy,
            "r": MultiSliceViewer._remove_xy,
        }

    @staticmethod
    def _process_key(event, key_bindings: dict) -> None:
        fig = event.canvas.figure
        ax = fig.axes[0]
        for key, method in key_bindings.items():
            if event.key == key:
                method(ax=ax)
        fig.canvas.draw()
        # fmt: off
        (fig.canvas.manager
        .set_window_title(f"Slice {ax.index}/{ax.volume.shape[0]}"))
        # fmt: on

    def _increment_slice(
        ax: matplotlib.axes.Axes,
        n: int,
        jump: bool = False,
    ) -> None:
        volume = ax.volume
        if jump:
            ax.index = (ax.index + volume.shape[0] // n) % volume.shape[0]
        else:
            ax.index = (ax.index + n) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])
        if ax.bboxes is not None:
            ax.red_foreground[..., 3] = ax.bboxes[ax.index] * ax.bbox_alpha
            ax.images[1].set_array(ax.red_foreground)

    _previous_slice = partial(_increment_slice, n=-1)
    _previous_slice = staticmethod(_previous_slice)
    _next_slice = partial(_increment_slice, n=1)
    _next_slice = staticmethod(_next_slice)
    _previous_jump = partial(_increment_slice, n=-10, jump=True)
    _previous_jump = staticmethod(_previous_jump)
    _next_jump = partial(_increment_slice, n=10, jump=True)
    _next_jump = staticmethod(_next_jump)

    # @staticmethod
    # def _previous_slice(ax: matplotlib.axes.Axes) -> None:
    #     volume = ax.volume
    #     ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    #     ax.images[0].set_array(volume[ax.index])
    #     if ax.bboxes is not None:
    #         ax.red_foreground[..., 3] = ax.bboxes[ax.index] * ax.bbox_alpha
    #         ax.images[1].set_array(ax.red_foreground)

    # @staticmethod
    # def _next_slice(ax: matplotlib.axes.Axes) -> None:
    #     volume = ax.volume
    #     ax.index = (ax.index + 1) % volume.shape[0]
    #     ax.images[0].set_array(volume[ax.index])
    #     if ax.bboxes is not None:
    #         ax.red_foreground[..., 3] = ax.bboxes[ax.index] * ax.bbox_alpha
    #         ax.images[1].set_array(ax.red_foreground)

    # @staticmethod
    # def _previous_jump(ax: matplotlib.axes.Axes) -> None:
    #     volume = ax.volume
    #     ax.index = (ax.index - volume.shape[0] // 10) % volume.shape[
    #         0
    #     ]  # wrap around using %
    #     ax.images[0].set_array(volume[ax.index])
    #     if ax.bboxes is not None:
    #         ax.red_foreground[..., 3] = ax.bboxes[ax.index] * ax.bbox_alpha
    #         ax.images[1].set_array(ax.red_foreground)

    # @staticmethod
    # def _next_jump(ax: matplotlib.axes.Axes):
    #     volume = ax.volume
    #     ax.index = (ax.index + volume.shape[0] // 10) % volume.shape[
    #         0
    #     ]  # wrap around using %
    #     ax.images[0].set_array(volume[ax.index])
    #     if ax.bboxes is not None:
    #         ax.red_foreground[..., 3] = ax.bboxes[ax.index] * ax.bbox_alpha
    #         ax.images[1].set_array(ax.red_foreground)

    @staticmethod
    def _select_rectangle() -> tuple[int, int, int, int]:
        corners = np.asarray(plt.ginput(2, timeout=-1)).astype("int")
        xmin = corners[:, 0].min()
        xmax = corners[:, 0].max()
        ymin = corners[:, 1].min()
        ymax = corners[:, 1].max()
        return xmin, xmax, ymin, ymax

    @staticmethod
    def _crop_xy(ax: matplotlib.axes.Axes) -> None:  # does not work with bboxes
        volume = ax.volume
        xmin, xmax, ymin, ymax = MultiSliceViewer._select_rectangle()
        ax.volume = volume[:, ymin:ymax, xmin:xmax]
        ax.images[0].set_array(ax.volume[ax.index])

    @staticmethod
    def _remove_xy(ax: matplotlib.axes.Axes) -> None:  # does not work with bboxes
        volume = ax.volume
        xmin, xmax, ymin, ymax = MultiSliceViewer._select_rectangle()
        volume[:, ymin:ymax, xmin:xmax] = volume.min()
        ax.volume = volume
        ax.images[0].set_array(ax.volume[ax.index])

    # def save_volume(ax):
    #     global volume
    #     volume = ax.volume

    @staticmethod
    def _remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith("keymap."):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)


if __name__ == "__main__":
    path = Path(askopenfilename())
    print(f"Openning '{path.name}'.")
    image_stack, metadata = load_image_stack(path=str(path))
    viewer = MultiSliceViewer(lognorm=True)
    viewer.plot(volume=image_stack)
    plt.show()
