from enum import StrEnum, auto
from typing import Protocol, runtime_checkable, Sequence

import numpy as np


@runtime_checkable  # optionnal
class Region(Protocol):
    """Protocol representing part of the scikit image RegionProperty
    class intended for functions that take arguments of that type.
    Such protocol is necessary because the RegionPropery class cannot be
    imported.
    """

    bbox: tuple


class Plane(StrEnum):
    """Class representing the three XY, XZ and YZ planes in a XYZ space.
    It is intended to facilitate dealing with slicing and axis specific
    operations when using numpy arrays.
    """

    XY = auto()
    XZ = auto()
    YZ = auto()

    @property
    def normal_axis(self) -> int:
        """Returns the index of the axis normal to the considered plane
        as used for the usual axis parameter in numpy functions.

        Returns:
            int: The axis index.
        """
        match self:
            case Plane.XY:
                index = 0
            case Plane.XZ:
                index = 1
            case Plane.YZ:
                index = 2
        return index

    def plane_selection_tuple(
        self,
        index: int | Sequence[int] | slice,
    ) -> tuple:
        """Returns the selection tuple to be used to access a plane at
        a specified a index with a numpy array. Also works with lists of
        indices and slices.

        Args:
            index (int | Sequence[int] | slice): The index of the
                selected plane along its normal axis.

        Returns:
            tuple: Selection tuple to be used in a numpy array.
        """
        match self:
            case Plane.XY:
                indices = (index, slice(None), slice(None))
            case Plane.XZ:
                indices = (slice(None), index, slice(None))
            case Plane.YZ:
                indices = (slice(None), slice(None), index)
        return indices

    def middle_slice(self, span: int) -> tuple:
        """Return the selection tuple for the middle plane of a numpy
        array from the number of layers of that plane type.

        Args:
            span (int): The number of layers in the array of that plane
                type.

        Returns:
            tuple: Selection tuple to be used in a numpy array to get
                its middle plane.
        """
        div, mod = divmod(span, 2)
        return self.plane_selection_tuple(index=div + mod)

    def crosses(self, region: Region, index: int) -> bool:
        """Returns True if the provided `Region` crosses the plane at
        the specified index.

        Args:
            region (Region): A `Region` object.
            index (int): The index of the layer that may cross the bbox
                of the `Region`.

        Returns:
            bool: Weither or not the `Region` crosses the plane at that
                index.
        """
        minz, miny, minx, maxz, maxy, maxx = region.bbox
        match self:
            case Plane.XY:
                if minz < index < maxz:
                    return True
                else:
                    return False
            case Plane.XZ:
                if miny < index < maxy:
                    return True
                else:
                    return False
            case Plane.YZ:
                if minx < index < maxx:
                    return True
                else:
                    return False
