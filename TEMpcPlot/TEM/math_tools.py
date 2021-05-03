

import numpy as np


def dist_p2vect(origin, vec, coor):
    """return the distances of a set of point from the line
       the position of the point if given by an iterable of shape 2xN
    """
    coor = np.array(coor)
    assert 2 in coor.shape
    if coor.shape[0] == 2:
        coor = coor.T
    return np.abs((np.cross(vec, coor - origin)) / mod(vec))


def perp_vect(vect):
    """return the perpendicular vector in a  2xN space
    """
    return np.cross(vect, [0, 0, 1])[:2] / mod(vect)


def mod(vect):
    return np.sqrt(vect.dot(vect))

