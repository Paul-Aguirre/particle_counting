from tkinter.filedialog import askopenfilename, askdirectory
from tkinter.messagebox import askyesno
from pathlib import Path
from functools import partial
from typing import Callable
from enum import Enum

import matplotlib.axes
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib
import toml

from files_io import load_image_stack, check_config


class RGBColorIndex(Enum):
    RED = (0,)
    GREEN = (1,)
    BLUE = (2,)
    YELLOW = 0, 1
    PURPLE = 0, 2
    CYAN = 1, 2
    WHITE = 0, 1, 2


def make_colored_overlay(
    volume: np.ndarray,
    alpha: float,
    color_index: tuple[int, int, int] | RGBColorIndex,
):
    bbox_overlay = np.zeros(
        (*volume.shape, 4),
        dtype=np.float32,
    )
    for i in color_index.value:
        bbox_overlay[..., i] = volume
    bbox_overlay[..., 3] = np.nan_to_num(volume / volume) * alpha
    # todo see if possible to silence zero division warning

    return bbox_overlay


class MultiSliceViewer:
    def __init__(
        self,
        lognorm: bool = False,
        cmap: str = "gray",
        datafile: Path | None = None,
    ) -> None:

        self.key_bindings = self._create_key_bindings()
        self._remove_keymap_conflicts(set(self.key_bindings.keys()))
        self.fig, self.ax = plt.subplots()
        self.lognorm = lognorm
        self.cmap = cmap
        if datafile:
            self.datafile: Path | None = Path(datafile)
        else:
            self.datafile: Path | None = None

        self.fig.canvas.mpl_connect(
            "key_press_event",
            partial(
                self._process_key,
                key_bindings=self.key_bindings,
            ),
        )

    def plot(
        self,
        volume: np.ndarray,
        overlay: np.ndarray | None = None,
    ):
        self.ax.volume = volume
        self.ax.index = volume.shape[0] // 2

        if self.lognorm:
            norm = colors.LogNorm(vmin=volume.min(), vmax=volume.max())
        else:
            norm = colors.Normalize()

        self.ax.imshow(volume[self.ax.index], norm=norm, cmap=self.cmap)

        if overlay is not None:
            if overlay.shape[0:3] != volume.shape:
                raise ValueError(
                    f"overlay (shape: {overlay.shape}) and volume "
                    f"(shape: {volume.shape}) must have compatible shapes."
                )

            self.update_overlay(overlay)

        else:
            self.ax.overlay = None

        self.fig.canvas.manager.set_window_title(
            f"Slice {self.ax.index}/{volume.shape[0]}"
        )

    def update_overlay(self, overlay):

        if (not hasattr(self.ax, "overlay")) or self.ax.overlay is None:
            self.ax.overlay = overlay
        else:
            self.ax.overlay[overlay != 0] = overlay[overlay != 0]

        self.ax.imshow(overlay[self.ax.index])

    @staticmethod
    def show() -> None:
        plt.show()

    @classmethod
    def display(cls, image_stack: np.ndarray, **kwargs) -> None:
        viewer = cls(**kwargs)
        viewer.plot(image_stack)
        viewer.show()
        return viewer

    @classmethod
    def display_file(cls, filepath: str | Path, **kwargs) -> None:
        image_stack, _ = load_image_stack(path=str(filepath))
        viewer = cls.display(image_stack, **kwargs)
        return viewer

    def save(self, file: str | Path):
        np.save(file, self.ax.volume)

    # @classmethod
    def _create_key_bindings(self) -> dict[str, Callable]:

        _previous_slice = partial(self._increment_slice, n=-1)
        _previous_slice = staticmethod(_previous_slice)
        _next_slice = partial(self._increment_slice, n=1)
        _next_slice = staticmethod(_next_slice)
        _previous_jump = partial(self._increment_slice, n=-10, jump=True)
        _previous_jump = staticmethod(_previous_jump)
        _next_jump = partial(self._increment_slice, n=10, jump=True)
        _next_jump = staticmethod(_next_jump)

        return {
            "j": _previous_slice,
            "k": _next_slice,
            "h": _previous_jump,
            "l": _next_jump,
            "c": self._crop_xy,
            "r": self._remove_xy,
            "s": self._save_volume,
            "z": self._select_substack,
        }

    def _process_key(self, event, key_bindings: dict) -> None:
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

    @staticmethod
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
        if ax.overlay is not None:
            ax.images[1].set_array(ax.overlay[ax.index])

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

    @staticmethod
    def _save_volume(ax: matplotlib.axes.Axes) -> None:
        if askyesno(message="Do you want to save this stack?"):
            filename = "volume.npy"
            np.save(filename, ax.volume)
            print(f'Stack saved as "{filename}" in {Path.cwd()}.')

    def _check_datafile_config(self) -> tuple[Path, Path, Path]:
        if not self.datafile:
            self.datafile = Path(
                askopenfilename(
                    title="Choose a datafile",
                )
            )
        return check_config(self.datafile)

    def _select_substack(self, ax: matplotlib.axes.Axes):

        xmin, xmax, ymin, ymax = self._select_rectangle()
        selection_coords = {
            "xstart": int(xmin),
            "xstop": int(xmax),
            "ystart": int(ymin),
            "ystop": int(ymax),
        }

        *_, configpath = self._check_datafile_config()
        with open(configpath, "r") as f:
            config = toml.load(f)
        if "selections" not in config.keys():
            config["selections"] = []
        config["selections"].append(selection_coords)
        with open(configpath, "w") as f:
            toml.dump(config, f)

        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                width=xmax - xmin,
                height=ymax - ymin,
                color="C2",
                zorder=1,
                fill=False,
            )
        )

    @staticmethod
    def _remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith("keymap."):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)


class MultiSliceViewerDuo(MultiSliceViewer):

    def __init__(
        self,
        lognorm: bool = False,
        cmap: str = "gray",
    ) -> None:

        self.key_bindings = self._create_key_bindings()
        self._remove_keymap_conflicts(set(self.key_bindings.keys()))
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.lognorm = lognorm
        self.cmap = cmap

        self.fig.canvas.mpl_connect(
            "key_press_event",
            partial(
                self._process_key,
                key_bindings=self.key_bindings,
            ),
        )

    def plot(
        self,
        volume1: np.ndarray,
        volume2: np.ndarray,
    ):
        assert volume1.shape == volume2.shape
        self.ax1.volume = volume1
        self.ax1.index = volume1.shape[0] // 2
        self.ax2.volume = volume2
        self.ax2.index = volume2.shape[0] // 2
        axes = self.ax1, self.ax2

        for ax in axes:
            if self.lognorm:
                norm = colors.LogNorm(
                    vmin=ax.volume.min(),
                    vmax=ax.volume.max(),
                )
            else:
                norm = colors.Normalize()

            ax.imshow(ax.volume[ax.index], norm=norm, cmap=self.cmap)

        self.fig.canvas.manager.set_window_title(
            f"Slice {self.ax1.index}/{volume1.shape[0]}"
        )

    @classmethod
    def display(
        cls,
        image_stack1: np.ndarray,
        image_stack2: np.ndarray,
        **kwargs,
    ) -> None:
        viewer = cls(**kwargs)
        viewer.plot(volume1=image_stack1, volume2=image_stack2)
        viewer.show()
        return viewer

    @classmethod
    def display_file():
        raise NotImplementedError

    def _increment_slice(
        ax1: matplotlib.axes.Axes,
        ax2: matplotlib.axes.Axes,
        n: int,
        jump: bool = False,
    ) -> None:

        axes = ax1, ax2
        volumes = ax1.volume, ax2.volume

        for ax, volume in zip(axes, volumes):
            if jump:
                ax.index = (ax.index + volume.shape[0] // n) % volume.shape[0]
            else:
                ax.index = (ax.index + n) % volume.shape[0]

            ax.images[0].set_array(volume[ax.index])

    @staticmethod
    def _process_key(event, key_bindings: dict) -> None:
        fig = event.canvas.figure
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        for key, method in key_bindings.items():
            if event.key == key:
                method(ax1=ax1, ax2=ax2)
        fig.canvas.draw()
        # fmt: off
        (fig.canvas.manager
        .set_window_title(f"Slice {ax1.index}/{ax1.volume.shape[0]}"))
        # fmt: on

    @staticmethod
    def _crop_xy(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _remove_xy(*args, **kwargs):
        raise NotImplementedError

    def _save_volume(*args, **kwargs):
        raise NotImplementedError


if __name__ == "__main__":
    # keep it without using the display_file method to have access
    # to image_stack and metadata after execution
    path = Path(askopenfilename())
    print(f"Openning '{path.name}'.")
    if path.suffix == ".nd2":
        image_stack, metadata = load_image_stack(path=str(path))
    elif path.suffix == ".npy":
        image_stack = np.load(path)
    else:
        raise ValueError('File must be valid "NPY" or "ND2" format.')
    viewer = MultiSliceViewer(lognorm=True)
    viewer.plot(volume=image_stack)
    viewer.show()
