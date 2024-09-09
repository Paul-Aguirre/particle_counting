from enum import StrEnum, auto
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable  # optionnal
class Region(Protocol):
    bbox: tuple


class Plane(StrEnum):
    XY = auto()
    XZ = auto()
    YZ = auto()

    @property
    def normal_axis(self):
        match self:
            case Plane.XY:
                index = 0
            case Plane.XZ:
                index = 1
            case Plane.YZ:
                index = 2
        return index

    def plane_selection_tuple(self, index: int | list[int]):
        match self:
            case Plane.XY:
                indices = (index, slice(None), slice(None))
            case Plane.XZ:
                indices = (slice(None), index, slice(None))
            case Plane.YZ:
                indices = (slice(None), slice(None), index)
            case _:
                raise ValueError
        return indices

    def middle_slice(self, span: int):
        # todo : does not need stack => only needs span (stack.shape[i])
        return self.plane_selection_tuple(index=span // 2)

    def crosses(self, region: Region, index: int) -> bool:
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
            case _:
                raise ValueError
