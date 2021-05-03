

import numpy as np


def dist_p2vert(verts, coor):
    """return the distances of a set of point from the line
       the position of the point if given by an iterable of shape 2xN
    """
    coor = np.array(coor)
    verts = np.array(verts)
    assert 2 in coor.shape
    if coor.shape[0] == 2:
        coor = coor.T
    vec = verts[1] - verts[0]
    return np.abs((np.cross(vec, coor - verts[0])) / (np.sqrt(vec.dot(vec))))
