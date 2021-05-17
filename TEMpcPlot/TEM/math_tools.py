'''a lot from gsas but still not in use
'''

import numpy as np
import numpy.linalg as nl
from scipy.spatial.transform import Rotation as R



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
    """
    modulus along axis 1
    """
    if vect.ndim == 1:
        return np.sqrt(vect @ vect)
    return np.sqrt(np.sum(np.power(vect, 2), axis=1))


def angle_between_vector(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    acos = (v1 @ v2) / (mod(v1) * mod(v2))
    return np.arccos(acos)


def find_common_peaks(tollerance, all_peaks):
    out = []
    # find common peaks, all_peaks has been shifted by the centers
    for i_p in all_peaks[0]:   # i_p one peak of the first image
        n_p = [i_p]
        for p_ima in all_peaks[1:]:

            dist = np.sqrt(np.sum((p_ima - i_p)**2, axis=1))
            if dist.min() > tollerance:
                break
            else:
                n_p.append(p_ima[dist.argmin()])
                i_p = p_ima[dist.argmin()]
        else:
            out.append(n_p)
    # out structure list of common peaks, each elem contains the position
    # of the peak for each image out.shape =  n_image,n_peaks,  2(x,y)
    return np.swapaxes(np.asanyarray(out), 0, 1)


def r2z(theta):
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, -sin], [sin, cos]]).T


def find_axis(tilts, rot=0):
    tilts = tilts[1:] - tilts[0]
    def creaxex(rot):
        axis = [R.from_rotvec([0, 0, rot]) * R.from_rotvec([i[0], 0, 0]) * R.from_rotvec([0, i[1], 0])  for i in tilts]
        rot_vec = [i.as_rotvec() for i in axis]
        rot_vec2 = [i / mod(i) for i in rot_vec]
        return rot_vec2


    rot_vec2 = creaxex(rot)
    print(*rot_vec2, sep='\n')
    axis = [R.from_rotvec([i[0], 0, 0]) *
            R.from_rotvec([0, i[1], 0]) *
            R.from_rotvec([0, 0, rot]) for i in tilts]
    rev = axis[0].inv()
    print([rev.apply(i) for i in rot_vec2])
    print('')

    #print(*rot_vec2, sep='\n')
    #print('')
    #print(*cr2.EwP._rot_vect, sep='\n')


# trig functions in degrees
def sind(x):
    return np.sin(x * np.pi / 180.)


def asind(x):
    return 180. * np.arcsin(x) / np.pi


def tand(x):
    return np.tan(x * np.pi / 180.)


def atand(x):
    return 180. * np.arctan(x) / np.pi


def atan2d(y, x):
    return 180. * np.arctan2(y, x) / np.pi


def cosd(x):
    return np.cos(x * np.pi / 180.)


def acosd(x):
    return 180. * np.arccos(x) / np.pi


def rdsq2d(x, p):
    return round(1.0 / np.sqrt(x), p)


rpd = np.pi / 180.
RSQ2PI = 1. / np.sqrt(2. * np.pi)
SQ2 = np.sqrt(2.)
RSQPI = 1. / np.sqrt(np.pi)
R2pisq = 1. / (2. * np.pi**2)
nxs = np.newaxis
