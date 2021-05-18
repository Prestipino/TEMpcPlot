'''a lot from gsas but still not in use
'''

import numpy as np
import numpy.linalg as nl
from scipy.spatial.transform import Rotation as R

from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

<<<<<<< HEAD
# trig functions in degrees
=======

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
>>>>>>> origin/3dRotation


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
<<<<<<< HEAD


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
    dot = v0 @ v1
    dot /= mod(v0) * mod(v1)
    return np.arccos(dot)


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


def rxyz(xyz, rot):
    return R.from_rotvec(np.insert([0., 0.], xyz, rot))


def defR(tilt, zrot):
    return rxyz(0, tilt[0]) * rxyz(1, tilt[1]) * rxyz(2, zrot)


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

################################################################################
def fillgmat(cell):
    """Compute lattice metric tensor from unit cell constants

    :param cell: tuple with a,b,c,alpha, beta, gamma (degrees)
    :return: 3x3 numpy array

    """
    a, b, c, alp, bet, gam = cell
    g = np.array([
        [a * a, a * b * cosd(gam), a * c * cosd(bet)],
        [a * b * cosd(gam), b * b, b * c * cosd(alp)],
        [a * c * cosd(bet), b * c * cosd(alp), c * c]])
    return g


def cell2Gmat(cell):
    """Compute real and reciprocal lattice metric tensor from unit cell constants

    :param cell: tuple with a,b,c,alpha, beta, gamma (degrees)
    :return: reciprocal (G) & real (g) metric tensors (list of two numpy 3x3 arrays)

    """
    g = fillgmat(cell)
    G = nl.inv(g)
    return G, g


def A2Gmat(A, inverse=True):
    """Fill real & reciprocal metric tensor (G) from A.

    :param A: reciprocal metric tensor elements as [G11,G22,G33,2*G12,2*G13,2*G23]
    :param bool inverse: if True return both G and g; else just G
    :return: reciprocal (G) & real (g) metric tensors (list of two numpy 3x3 arrays)

    """
    G = np.array([
        [A[0], A[3] / 2., A[4] / 2.],
        [A[3] / 2., A[1], A[5] / 2.],
        [A[4] / 2., A[5] / 2., A[2]]])
    if inverse:
        g = nl.inv(G)
        return G, g
    else:
        return G


def Gmat2A(G):
    """Extract A from reciprocal metric tensor (G)

    :param G: reciprocal maetric tensor (3x3 numpy array
    :return: A = [G11,G22,G33,2*G12,2*G13,2*G23]

    """
    return [G[0][0], G[1][1], G[2][2], 2. * G[0][1], 2. * G[0][2], 2. * G[1][2]]


def cell2A(cell):
    """Obtain A = [G11,G22,G33,2*G12,2*G13,2*G23] from lattice parameters

    :param cell: [a,b,c,alpha,beta,gamma] (degrees)
    :return: G reciprocal metric tensor as 3x3 numpy array

    """
    G, g = cell2Gmat(cell)
    return Gmat2A(G)


def A2cell(A):
    """Compute unit cell constants from A

    :param A: [G11,G22,G33,2*G12,2*G13,2*G23] G - reciprocal metric tensor
    :return: a,b,c,alpha, beta, gamma (degrees) - lattice parameters

    """
    G, g = A2Gmat(A)
    return Gmat2cell(g)


def Gmat2cell(g):
    """Compute real/reciprocal lattice parameters from real/reciprocal metric tensor (g/G)
    The math works the same either way.

    :param g (or G): real (or reciprocal) metric tensor 3x3 array
    :return: a,b,c,alpha, beta, gamma (degrees) (or a*,b*,c*,alpha*,beta*,gamma* degrees)

    """
    oldset = np.seterr('raise')
    a = np.sqrt(max(0, g[0][0]))
    b = np.sqrt(max(0, g[1][1]))
    c = np.sqrt(max(0, g[2][2]))
    alp = acosd(g[2][1] / (b * c))
    bet = acosd(g[2][0] / (a * c))
    gam = acosd(g[0][1] / (a * b))
    np.seterr(**oldset)
    return a, b, c, alp, bet, gam


def invcell2Gmat(invcell):
    """Compute real and reciprocal lattice metric tensor from reciprocal 
       unit cell constants

    :param invcell: [a*,b*,c*,alpha*, beta*, gamma*] (degrees)
    :return: reciprocal (G) & real (g) metric tensors (list of two 3x3 arrays)

    """
    G = fillgmat(invcell)
    g = nl.inv(G)
    return G, g


def prodMGMT(G, Mat):
    '''Transform metric tensor by matrix

    :param G: array metric tensor
    :param Mat: array transformation matrix
    :return: array new metric tensor

    '''
    return np.inner(np.inner(Mat, G), Mat)  # right
#    return np.inner(Mat,np.inner(Mat,G))       #right
#    return np.inner(np.inner(G,Mat).T,Mat)      #right
#    return np.inner(Mat,np.inner(G,Mat).T)     #right


def TransformCell(cell, Trans):
    '''Transform lattice parameters by matrix

    :param cell: list a,b,c,alpha,beta,gamma,(volume)
    :param Trans: array transformation matrix
    :return: array transformed a,b,c,alpha,beta,gamma,volume

    '''
    newCell = np.zeros(7)
    g = cell2Gmat(cell)[1]
    newg = prodMGMT(g, Trans)
    newCell[:6] = Gmat2cell(newg)
    newCell[6] = calc_V(cell2A(newCell[:6]))
    return newCell


def calc_rVsq(A):
    """Compute the square of the reciprocal lattice volume (1/V**2) from A'

    """
    G, g = A2Gmat(A)
    rVsq = nl.det(G)
    if rVsq < 0:
        return 1
    return rVsq


def calc_rV(A):
    """Compute the reciprocal lattice volume (V*) from A
    """
    return np.sqrt(calc_rVsq(A))


def calc_V(A):
    """Compute the real lattice volume (V) from A
    """
    return 1. / calc_rV(A)


def A2invcell(A):
    """Compute reciprocal unit cell constants from A
    returns tuple with a*,b*,c*,alpha*, beta*, gamma* (degrees)
    """
    G, g = A2Gmat(A)
    return Gmat2cell(G)


def Gmat2AB(G):
    """Computes orthogonalization matrix from reciprocal metric tensor G

    :returns: tuple of two 3x3 numpy arrays (A,B)

       * A for crystal to Cartesian transformations (A*x = np.inner(A,x) = X)
       * B (= inverse of A) for Cartesian to crystal transformation (B*X = np.inner(B,X) = x)

    """
#    cellstar = Gmat2cell(G)
    g = nl.inv(G)
    cell = Gmat2cell(g)
#    A = np.zeros(shape=(3,3))
    return cell2AB(cell)
#    # from Giacovazzo (Fundamentals 2nd Ed.) p.75
#    A[0][0] = cell[0]                # a
#    A[0][1] = cell[1]*cosd(cell[5])  # b cos(gamma)
#    A[0][2] = cell[2]*cosd(cell[4])  # c cos(beta)
#    A[1][1] = cell[1]*sind(cell[5])  # b sin(gamma)
#    A[1][2] = -cell[2]*cosd(cellstar[3])*sind(cell[4]) # - c cos(alpha*) sin(beta)
#    A[2][2] = 1./cellstar[2]         # 1/c*
#    B = nl.inv(A)
#    return A,B


def cell2AB(cell):
    """Computes orthogonalization matrix from unit cell constants

    :param tuple cell: a,b,c, alpha, beta, gamma (degrees)
    :returns: tuple of two 3x3 numpy arrays (A,B)
       A for crystal to Cartesian transformations A*x = np.inner(A,x) = X 
       B (= inverse of A) for Cartesian to crystal transformation B*X = np.inner(B,X) = x
    """
    G, g = cell2Gmat(cell)
    cellstar = Gmat2cell(G)
    A = np.zeros(shape=(3, 3))
    # from Giacovazzo (Fundamentals 2nd Ed.) p.75
    A[0][0] = cell[0]                # a
    A[0][1] = cell[1] * cosd(cell[5])  # b cos(gamma)
    A[0][2] = cell[2] * cosd(cell[4])  # c cos(beta)
    A[1][1] = cell[1] * sind(cell[5])  # b sin(gamma)
    # - c cos(alpha*) sin(beta)
    A[1][2] = -cell[2] * cosd(cellstar[3]) * sind(cell[4])
    A[2][2] = 1. / cellstar[2]         # 1/c*
    B = nl.inv(A)
    return A, B


def Rh2Hx(Rh):
    'rhombohedral to hexagonal conversion'
    Hx = [0, 0, 0]
    Hx[0] = Rh[0] - Rh[1]
    Hx[1] = Rh[1] - Rh[2]
    Hx[2] = np.sum(Rh)
    return Hx


def Hx2Rh(Hx):
    'hexagonal to rhombohedral conversion'
    Rh = [0, 0, 0]
    itk = -Hx[0] + Hx[1] + Hx[2]
    if itk % 3 != 0:
        return 0  # error - not rhombohedral reflection
    else:
        Rh[1] = itk // 3
        Rh[0] = Rh[1] + Hx[0]
        Rh[2] = Rh[1] - Hx[1]
        if Rh[0] < 0:
            for i in range(3):
                Rh[i] = -Rh[i]
        return Rh
=======
>>>>>>> origin/3dRotation
