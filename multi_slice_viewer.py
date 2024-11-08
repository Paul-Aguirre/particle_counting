"""This module mainly implements the `MultiSliceViewer` class which is
an interface that provides slice by slice visualisation and some basic
manipulations for image stacks.

The user interacts with the interface is done via key bindings and mouse
clicks.

`MultiSliceViewerDuo` is a less advanced sibling class which allows
visualisation of two image stacks, intended for comparison purposes.

`make_colored_overlay()` is a function intended to create overlays from
image stacks. These overlays can also be used for comparison purposes in
the base `MultiSliceViewer` class.

If the module is run as a script, it will prompt the user to select
a file. It can only take in *ND2* and *NPY* files. The file will then be
rendered with the `MultiSliceViewer` class.
"""

from tkinter.filedialog import askopenfilename
from tkinter.messagebox import askyesno
from pathlib import Path
from functools import partial
from typing import Callable
from enum import Enum
import argparse
import warnings

import matplotlib.axes
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib
import toml

from files_io import get_save_path, load_image_stack, check_config


class MultiSliceViewer:
    """This class implements an interface that provides slice by slice
    visualisation and some basic manipulations on image stacks. It is
    based on the pyplot module from the matplotlib library.

    The user interacts with the interface is done via key bindings and
    mouse clicks. The current slice number and the total number of
    slices is display as the window title.

    The key bindings are listed below:

    "j"
    :   Displays the previous slice.

    "k"
    :   Displays the next slice.

    "h"
    :   Makes a jump to the slice 1/10 of the total number of slices
        before the current slice.

    "l"
    :   Makes a jump to the slice 1/10 of the total number of slices
        after the current slice.

    "c"
    :   Triggers *crop* mode. In this mode, the user has to select a
        rectangular region by clicking twice on the current slice,
        defining two corners. Everything outside the selection is then
        cropped out of the image. This results in an image stack in
        which the images are all cropped to the dimension of the
        selected region. This operation is **irreversible** for the
        currently loaded data. The image stack will have to be reloaded
        to be visible again.

    "r"
    :   Trigger *remove* mode. In this mode, the user has to select a
        rectangular region by clicking twice on the current slice,
        defining two corners. Everything inside the selection is then
        cropped out of the image, the pixel values set to 0 throughout
        the image stack.
        This results in an image stack of the same dimensions but with
        the same cropped out region on each slice.
        This operation is **irreversible** for the currently loaded
        data. The image stack will have to be reloaded to be visible
        again.

    "s"
    :   Asks the user if they want to save the current image stack
        (taking cropping and removals into account), and saves it as
        an NPY file in the adequate directory if the datafile attribute
        was supplied to the class constructor or else in the current
        working directory as "volume.npy".

    "z"
    :   Trigger *selection* mode. In this mode, the user has to select a
        rectangular region by clicking twice on the current slice,
        defining two corners. The x and y coordinates of the selection
        are then saved in the configuration file corresponding to
        the supplied datafile.
        This operation is *irreversible* in the sense that
        faulty selection data will have to be removed manually from
        the configuration file once registered.

    """

    def __init__(
        self,
        lognorm: bool = False,
        cmap: str = "gray",
        datafile: Path | None = None,
    ) -> None:

        self._key_bindings = self._create_key_bindings()
        self._remove_keymap_conflicts(set(self._key_bindings.keys()))
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
                key_bindings=self._key_bindings,
            ),
        )

    def plot(
        self,
        volume: np.ndarray,
        overlay: np.ndarray | None = None,
    ) -> None:
        """Plots the image stack provided in volume argument starting
        from the middle slice with the overlay on top of it,
        if provided. The stack is plotted starting from central slice.

        Args:
            volume (np.ndarray): A grayscale image stack represented by
                a numpy array.
            overlay (np.ndarray | None, optional): An RGBA image stack
                represented by a numpy array, for instance generated by
                a call to `make_colored_overlay()`. Defaults to None.

        Raises:
            ValueError: When the shapes of the overlay and of the
                image stack are not compatible.
        """
        self.ax.volume = volume
        self.ax.index = volume.shape[0] // 2

        if self.lognorm:
            norm = colors.LogNorm(vmin=volume.min(), vmax=volume.max())
        else:
            norm = colors.Normalize()

        self.ax.imshow(volume[self.ax.index], norm=norm, cmap=self.cmap)

        if overlay is not None:
            if overlay.shape[0:3] != volume.shape:
                # because overlay is an RGBA image stack
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

    def update_overlay(self, overlay: np.ndarray) -> None:
        """If the `self.ax.overlay` was not previously defined or was
        defined to None, this method defines it as provided in its
        overlay argument.

        Otherwise, it attributes new values taken from
        the overlay argument only where the pre-existing self.ax.overlay
        had zero values.

        Args:
            overlay (np.ndarray): An overlay array with dimentions
                compatible with `self.ax.volume`, for instance provided
                by the `make_colored_overlay()` function.
        """

        if (not hasattr(self.ax, "overlay")) or self.ax.overlay is None:
            self.ax.overlay = overlay
        else:
            self.ax.overlay[overlay != 0] = overlay[overlay != 0]

        self.ax.imshow(overlay[self.ax.index])

    @staticmethod
    def show() -> None:
        """A simple convinience method that just calls pyplot.show().
        This removes the need to import matplotlib.pyplot in a script
        just to be able to call the show() method in order to open the
        viewer.
        """
        plt.show()

    @classmethod
    def display[
        T: MultiSliceViewer,
    ](
        cls: T,
        image_stack: np.ndarray,
        overlay: np.ndarray | None = None,
        **kwargs,
    ) -> T:
        """Convinience class method to quickly plot and display an image
        stack in one function call.

        Args:
            image_stack (np.ndarray): An array representing a stack of
                grayscale or binary images.
            kwargs: All additionnal keword arguments are passed down to
                the class constructor.

        Returns:
            MultiSliceViewer: The MultiSliceViewer instance created
                to display the provided image_stack.
        """
        viewer = cls(**kwargs)
        viewer.plot(image_stack, overlay)
        viewer.show()
        return viewer

    @classmethod
    def display_file[
        T: MultiSliceViewer
    ](cls: T, filepath: str | Path, **kwargs,) -> T:
        """Convinience class method similar to `display` but takes
        a path to a data file instead of the image stack data itself.

        Args:
            filepath (str | Path): A path-like object pointing to
                an image stack data file.
            kwargs: All additionnal keword arguments are passed down to
                the class constructor.

        Returns:
            MultiSliceViewer: The MultiSliceViewer instance created
                to display the provided image_stack.
        """
        image_stack, _ = load_image_stack(path=str(filepath))
        kwargs["datafile"] = filepath
        # (to provide the data file path to the __init__ method)
        viewer = cls.display(image_stack, **kwargs)
        return viewer

    def save(self, file: str | Path):
        """Saves the current content of the stack (cropping taken into
        account) as NPY save file, through a call call to numpy's save
        function.

        Args:
            file (str | Path): A path-like object pointing at where
                to save the file.
        """
        np.save(file, self.ax.volume)

    def show_selections(self) -> None:
        *_, configpath = self._check_datafile_config()
        with open(configpath, "r") as f:
            config = toml.load(f)
        if config["selections"] == []:
            print("No selection to show.")
        else:
            self._show_rectangle_from_coords_list(
                list=config["selections"],
                fill=False,
                color="C2",
                zorder=1,
            )

    def show_patches(self, patches) -> None:
        self._show_rectangle_from_coords_list(
            list=patches,
            fill=True,
            color="C3",
            zorder=1,
            alpha=0.5,
        )

    def show_patches_all(self) -> None:
        *_, configpath = self._check_datafile_config()
        with open(configpath, "r") as f:
            config = toml.load(f)
        if config["selections"] == []:
            print("No selection to show.")
        else:
            for i, selection in enumerate(config["selections"]):
                if "patches" in selection.keys():
                    self.show_patches(selection["patches"])
                else:
                    print(f"No patches to show in selection no.{i}.")

    @staticmethod
    def _remove_keymap_conflicts(new_keys_set: set):
        """Inspects pyplot's default rcParams for key binding, compare
        them with new key bindings and removes the old ones in case of
        a conflict with a new one.

        Args:
            new_keys_set (set): A set of the new keys used for
                the custom key bindings.
        """
        for prop in plt.rcParams:
            if prop.startswith("keymap."):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    # @classmethod
    def _create_key_bindings(self) -> dict[str, Callable]:
        """Creates and returns the methods used to operate the viewer
        through the key bindings.

        Returns:
            dict[str, Callable]: Contains the methods for the key
            bindings as values and the keys to be bound as keys.
        """

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
            "p": self._add_patch,
        }

    def _process_key(self, event, key_bindings: dict) -> None:
        """Takes care of calling the appropirate method when a bound key
        is pressed.
        """
        fig = event.canvas.figure
        ax = fig.axes[0]
        for key, method in key_bindings.items():
            if event.key == key:
                method()
        fig.canvas.draw()
        # fmt: off
        (fig.canvas.manager
        .set_window_title(f"Slice {ax.index}/{ax.volume.shape[0]}"))
        # fmt: on

    def _increment_slice(
        self,
        n: int,
        jump: bool = False,
    ) -> None:
        """Generic method for moving up and down in an image stack.
        It does so by incrementing the index of the currently displayed
        slice.

        Is used internally to create methods present in key bindings.

        Args:
            ax (matplotlib.axes.Axes): target matplotlib.axes.Axes object
            n (int): Represents the increment in slice number.
            jump (bool, optional): If True, `n` is instead interpreted
                as the fraction of total number of slices to be jumped.
                eg. if `jump == True` and `n==5`, this method will
                increment the index of the currently displayed slice
                by 1/5 of the total number of slices. Defaults to False.
        """
        volume = self.ax.volume
        if jump:
            self.ax.index = (self.ax.index + volume.shape[0] // n) % volume.shape[0]
        else:
            self.ax.index = (self.ax.index + n) % volume.shape[0]
        self.ax.images[0].set_array(volume[self.ax.index])
        if self.ax.overlay is not None:
            self.ax.images[1].set_array(self.ax.overlay[self.ax.index])

    @staticmethod
    def _select_rectangle() -> tuple[int, int, int, int]:
        """Uses pyplot.ginput function to return rectagular selection
        coordinates.

        Returns:
            tuple[int, int, int, int]: Rectangular selection
                coordinates, respectively xmin, xmax, ymin and ymax.
        """
        corners = np.asarray(plt.ginput(2, timeout=-1)).astype("int")
        xmin = corners[:, 0].min()
        xmax = corners[:, 0].max()
        ymin = corners[:, 1].min()
        ymax = corners[:, 1].max()
        return xmin, xmax, ymin, ymax

    def _choose_selection(self, selections) -> int | Warning:
        point = np.asarray(plt.ginput(1, timeout=-1)[0]).astype(np.uint16)
        for i, selection in enumerate(selections):
            if (
                selection["xstart"] <= point[0] < selection["xstop"]
                and selection["ystart"] <= point[1] < selection["ystop"]
            ):
                return i
        return warnings.warn(
            message=f"No selection containing that point ({point}) was found.",
        )

    @staticmethod
    def _crop_xy(ax: matplotlib.axes.Axes) -> None:  # does not work with bboxes
        """Crops the current image stack given a rectangle selected in
        the xy plane.

        Args:
            ax (matplotlib.axes.Axes): The target matplotlib.axes.Axes
                object.
        """
        volume = ax.volume
        xmin, xmax, ymin, ymax = MultiSliceViewer._select_rectangle()
        ax.volume = volume[:, ymin:ymax, xmin:xmax]
        ax.images[0].set_array(ax.volume[ax.index])

    @staticmethod
    def _remove_xy(ax: matplotlib.axes.Axes) -> None:  # does not work with bboxes
        """Sets the pixel values to 0 in a region of rectangular section
        selected in the xy plane.

        Args:
            ax (matplotlib.axes.Axes): The target matplotlib.axes.Axes
                object.
        """
        volume = ax.volume
        xmin, xmax, ymin, ymax = MultiSliceViewer._select_rectangle()
        volume[:, ymin:ymax, xmin:xmax] = volume.min()
        ax.volume = volume
        ax.images[0].set_array(ax.volume[ax.index])

    def _patch_xy(self, selection: dict) -> None:
        xmin, xmax, ymin, ymax = self._select_rectangle()

        # limiting the patch size to the selection boundary:
        xmin = int(max(xmin, selection["xstart"]))
        xmax = int(min(xmax, selection["xstop"]))
        ymin = int(max(ymin, selection["ystart"]))
        ymax = int(min(ymax, selection["ystop"]))

        if "patches" not in selection.keys():
            selection["patches"] = []
        selection["patches"].append(
            {
                "xstart": xmin,
                "xstop": xmax,
                "ystart": ymin,
                "ystop": ymax,
            }
        )
        self.ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                width=xmax - xmin,
                height=ymax - ymin,
                color="C3",
                zorder=0.9,
                fill=True,
                alpha=0.5,
            )
        )

    def _add_patch(self) -> None:
        *_, configpath = self._check_datafile_config()
        with open(configpath, "r") as f:
            config = toml.load(f)
        i = self._choose_selection(config["selections"])
        if isinstance(i, int):
            selection = config["selections"][i]
            self._patch_xy(selection)
            with open(configpath, "w") as f:
                toml.dump(config, f)
        elif isinstance(i, Warning):
            raise i

    def _save_volume(self) -> None:
        """Asks the user if they want to save the current image stack
        (taking cropping into account), and saves it as an NPY file in
        the adequate directory if the datafile attribute was supplied
        to the class constructor or else in the current working
        directory as "volume.npy".
        """
        if askyesno(message="Do you want to save this stack?"):
            if self.datafile:
                filename = get_save_path(
                    datapath=self.datafile,
                    suffix="volume",
                    ext="npy",
                )
            else:
                filename = "volume.npy"
            np.save(filename, self.ax.volume)
            print(f'Stack saved as "{filename}" in {Path.cwd()}.')

    def _check_datafile_config(self) -> tuple[Path, Path, Path]:
        """Checks if `self.datafile` attribute was provided and prompts
        the user to choose a data file.

        Returns:
            tuple[Path, Path, Path]: Three Path objects pointing to
                (respectively): the datafile, the related directory,
                the related configuration file
        """
        if not self.datafile:
            self.datafile = Path(
                askopenfilename(
                    title="No datafile registered, "
                    "please choose the correct datafile",
                )
            )
        return check_config(self.datafile)

    def _select_substack(self):
        """Allows the user to select a region of an image stack in
        the xy plane. The coordinates for that region are then saved in
        the configuration file corresponding to the currently displayed
        file. Prompts the user to choose a data file if it was not
        provided to the class constructor.
        """

        xmin, xmax, ymin, ymax = self._select_rectangle()
        selection_coords = {
            "xstart": int(xmin),
            "xstop": int(xmax),
            "ystart": int(ymin),
            "ystop": int(ymax),
            "zslice": self.ax.index,
        }

        *_, configpath = self._check_datafile_config()
        with open(configpath, "r") as f:
            config = toml.load(f)
        if "selections" not in config.keys():
            config["selections"] = []
        config["selections"].append(selection_coords)
        with open(configpath, "w") as f:
            toml.dump(config, f)

        # adding rectangle to selected region through matplotlib instead
        # of the overlay system because it proved to be not very
        # distinguishable on very large images.
        self.ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                width=xmax - xmin,
                height=ymax - ymin,
                color="C2",
                zorder=1,
                fill=False,
            )
        )

    def _show_rectangle_from_coords_list(
        self,
        list: list,
        fill: bool = False,
        color: str | tuple = "C0",
        zorder: float | int = 1,
        alpha: float = 1.0,
    ) -> None:
        for coords in list:
            xmin = coords["xstart"]
            xmax = coords["xstop"]
            ymin = coords["ystart"]
            ymax = coords["ystop"]

            self.ax.add_patch(
                plt.Rectangle(
                    (xmin, ymin),
                    width=xmax - xmin,
                    height=ymax - ymin,
                    color=color,
                    zorder=zorder,
                    fill=fill,
                    alpha=alpha,
                )
            )


class MultiSliceViewerDuo(MultiSliceViewer):
    """A class that closely resembles the MultiSliceViewer class but is
    meant to visualise two image stacks side by side.

    Although, some functionnalities from the parent class have been
    deleted because they do not make a lot a sense in the context of
    this class.
    """

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
        self,
        n: int,
        jump: bool = False,
    ) -> None:

        axes = self.ax1, self.ax2
        volumes = self.ax1.volume, self.ax2.volume

        for ax, volume in zip(axes, volumes):
            if jump:
                ax.index = (ax.index + volume.shape[0] // n) % volume.shape[0]
            else:
                ax.index = (ax.index + n) % volume.shape[0]

            ax.images[0].set_array(volume[ax.index])

    @staticmethod
    def _crop_xy(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _remove_xy(*args, **kwargs):
        raise NotImplementedError

    def _save_volume(*args, **kwargs):
        raise NotImplementedError


class RGBColorIndex(Enum):
    """Enum representing the index for each color in a numpy array
    representing an RGB image. Used in `make_colored_overlay()`.

    The indices are stored as tuples to allow for more colors.
    """

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
) -> np.ndarray:
    """Returns an RGBA version of `volume` with the supplied `alpha`
    value and color. Used to make overlays from images containing
    bboxes, thresholded images

    This function is intended to supply the overlay parameter to the
    plot() method from the MultiSliceViewer class.

    Args:
        volume (np.ndarray): The base binary image for the overlay.
        alpha (float): The alpha value for the overlay.
        color_index (tuple[int, int, int] | RGBColorIndex): A tuple
            specifying which of the R, G and B colors should be used in
            the overlay.

    Returns:
        np.ndarray: The RGBA array version of volume that can be used
            as an overlay.
    """
    bbox_overlay = np.zeros(
        (*volume.shape, 4),
        dtype=np.float32,
    )
    for i in color_index.value:
        bbox_overlay[..., i] = volume
    bbox_overlay[..., 3] = np.nan_to_num(volume / volume) * alpha
    # this puts a nonzero alpha value everywhere where the initial array
    # is nonzero
    # todo see if possible to silence zero division warning

    return bbox_overlay


def main():
    """Prompts the user to choose a file to visualise.

    Raises:
        ValueError: In case the selected file is not of the right type.
    """
    parser = argparse.ArgumentParser(prog="multi_slice_viewer")
    parser.add_argument(
        "-d",
        "--datapath",
        action="store",
        required=False,
        default=None,
        type=Path,
        help="The path to the datafile to display.",
    )
    parser.add_argument(
        "-r",
        "--reader",
        action="store",
        required=False,
        choices=["nd2", "nd2reader"],
        default="nd2reader",
        help="The reader to use for openning ND2 files. 'nd2'"
        " option is necessarry for 8-bit files.",
    )
    # fmt: off
    parser.add_argument(
        "--lognorm",
        action="store_true",
        help="Weither to use a logarithmic norm or not for "
        "the displayed images.",
    )
    # fmt: on
    parser.add_argument(
        "--hide_selections",
        action="store_true",
        help="Weither to hide the selected areas.",
    )
    parser.add_argument(
        "--show_all_patches",
        action="store_true",
        help="Weither to show all patched areas.",
    )
    args = parser.parse_args()

    if args.datapath is None:
        args.datapath = Path(askopenfilename())
    if args.datapath.suffix == ".nd2":
        image_stack, _ = load_image_stack(
            path=str(args.datapath),
            reader=args.reader,
        )
    elif args.datapath.suffix == ".npy":
        image_stack = np.load(args.datapath)
    else:
        raise ValueError('File must be valid "NPY" or "ND2" format.')
    print(f"Openning '{args.datapath.name}'.")
    viewer = MultiSliceViewer(
        lognorm=args.lognorm,
        datafile=args.datapath,
    )
    viewer.plot(volume=image_stack)
    show_selections = not args.hide_selections
    if show_selections:
        viewer.show_selections()
    if args.show_all_patches:
        viewer.show_patches_all()
    viewer.show()


if __name__ == "__main__":
    main()
