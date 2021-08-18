import numpy as np

from .bp2DRct import Box
from .bp2DPnt import Point


def name_rectangles(rcts):
    for i, r in enumerate(rcts):
        r.n = i
    return rcts

# @staticmethod
def area_of_rectangles(rcts):
    return np.sum([rct.a for rct in rcts]) if rcts else 0


# @staticmethod
def x_extension_of_rectangles(rcts):
    return np.max([rct.tr.get_x() for rct in rcts]) if rcts else 0


# @staticmethod
def y_extension_of_rectangles(rcts):
    return np.max([rct.tr.get_y() for rct in rcts]) if rcts else 0


# @staticmethod
def bounding_box_of_rectangles(rcts):
    if not rcts:
        return Box(0, 0)
    xbl = np.min([rct.bl.get_x() for rct in rcts])
    ybl = np.min([rct.bl.get_y() for rct in rcts])
    xtr = np.max([rct.tr.get_x() for rct in rcts])
    ytr = np.max([rct.tr.get_y() for rct in rcts])
    return Box(xtr - xbl, ytr - ybl, Point(xbl, ybl))


# @staticmethod
def sort_rectangles(rcts):
    def get_a(rct):
        return rct.a
    return (sorted(rcts, key=get_a)[::-1])