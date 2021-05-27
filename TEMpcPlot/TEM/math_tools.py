'''a lot from gsas but still not in use
'''

import numpy as np
# import numpy.linalg as nl
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares


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


rpd = np.pi / 180.
RSQ2PI = 1. / np.sqrt(2. * np.pi)
SQ2 = np.sqrt(2.)


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


def norm(vect):
    return np.array(vect, dtype=float) / mod(vect)


def angled_between_tilts(x1, y1, x2, y2):
    """
    angle between two planes from the tilts
    """
    return acosd(cosd(x1) * cosd(y1) * cosd(x2) * cosd(y2) +
                 cosd(x1) * sind(y1) * cosd(x2) * sind(y2) +
                 sind(x1) * sind(x2))


def angle_between_tilts(x1, y1, x2, y2):
    """
    angle between two planes from the tilts
    """
    return np.arccos(np.cos(x1) * np.cos(y1) * np.cos(x2) * np.cos(y2) +
                     np.cos(x1) * np.sin(y1) * np.cos(x2) * np.sin(y2) +
                     np.sin(x1) * np.sin(x2))


def angle_between_vectors(v0, v1):
    """Return angle between vectors.
    If directed is False, the input vectors are interpreted as undirected axes,
    i.e. the maximum angle is pi/2.
    >>> a = angle_between_vectors([1, -2, 3], [-1, 2, -3])
    >>> numpy.allclose(a, math.pi)
    True
    """
    v0 = np.array(v0, dtype=np.float64, copy=False)
    v1 = np.array(v1, dtype=np.float64, copy=False)
    s = mod(np.cross(v0, v1))
    c = v0 @ v1
    return np.arctan2(s, c)


def zrot_among_vectors(v0, v1):
    """Return angle between vectors layng on the xy plane.
    If directed is False, the input vectors are interpreted as undirected axes,
    i.e. the maximum angle is pi/2.
    >>> a = angle_between_vectors([1, -2, 3], [-1, 2, -3])
    >>> numpy.allclose(a, math.pi)
    True
    """
    v0 = np.array(v0, dtype=np.float64, copy=False)
    v1 = np.array(v1, dtype=np.float64, copy=False)
    s = np.cross(v0, v1)
    c = v0 @ v1
    segno = np.sign(s.flatten()[2::3])
    return segno * np.arctan2(mod(s), c)


def find_common_peaks(tollerance, all_peaks):
    """
        # list of array n*2 or 3 
        # out structure list of common peaks, each elem contains the position
        # of the peak for each image out.shape =  n_image,n_peaks,  2(x,y)
    """
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


def zrotm(theta):
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, -sin], [sin, cos]]).T


def rotxyz(x, y, z):
    rz = R.from_rotvec([0, 0, z])
    rx = R.from_rotvec([0, x, 0])
    ry = R.from_rotvec([y, 0, 0])
    return rz * rx * ry


def creaxex(tilts, zrot):
    """
    create the rotation vector based on tilt and camera rotation
    """
    r0 = rotxyz(tilts[0][0], tilts[0][1], zrot)
    r0i = r0.inv()
    axis = [rotxyz(i[0], i[1], zrot) for i in tilts[1:]]
    rot_vec = [i * r0i for i in axis]
    rot_vec2 = [i.as_rotvec() for i in rot_vec]
    return rot_vec2


def find_absolute_angle(tiltin, axis, zrot=None):
    """
    """
    def drot_axes(zrot):
        rot_axes = [defR(tl, zrot) for tl in tiltin]
        rinv = defR(tiltin[0], zrot).inv() 
        rot_axes = [rinv * r for r in rot_axes]
        rot_axes = [r.as_rotvec() for r in rot_axes]
        print(*rot_axes, sep='\n')
        return np.array([r / mod(r) for r in rot_axes])

    def resid(zrot):
        rot_axes = drot_axes(zrot)
        return np.sum(np.abs(rot_axes) - np.abs(axis))

    print(*np.round(drot_axes(zrot), 3), sep='\n')
    print('\n')
    #res_1 = least_squares(resid, zrot, verbose=1) 
    print(axis, '\n')
    #print(*np.round(drot_axes(res_1.x[0]) , 3), sep='\n')
    #return res_1.x
    return resid(zrot)


################################################################################
def sec2HMS(sec):
    """Convert time in sec to H:M:S string

    :param sec: time in seconds
    :return: H:M:S string (to nearest 100th second)

    """
    H = int(sec // 3600)
    M = int(sec // 60 - H * 60)
    S = sec - 3600 * H - 60 * M
    return '%d:%2d:%.2f' % (H, M, S)

###############################################################################
