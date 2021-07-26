import numpy as np
import numpy.linalg as nl
from . import math_tools as mt
from .math_tools import sind, cosd, acosd
from itertools import combinations
from itertools import product as nestedLoop


rpd = np.pi / 180.
RSQ2PI = 1. / np.sqrt(2. * np.pi)
SQ2 = np.sqrt(2.)
RSQPI = 1. / np.sqrt(np.pi)
R2pisq = 1. / (2. * np.pi**2)
nxs = np.newaxis


def sec2HMS(sec):
    """Convert time in sec to H:M:S string

    :param sec: time in seconds
    :return: H:M:S string (to nearest 100th second)

    """
    H = int(sec // 3600)
    M = int(sec // 60 - H * 60)
    S = sec - 3600 * H - 60 * M
    return '%d:%2d:%.2f' % (H, M, S)


def rotdMat(angle, axis=0):
    """Prepare rotation matrix for angle in degrees about axis(=0,1,2)

    :param angle: angle in degrees
    :param axis:  axis (0,1,2 = x,y,z) about which for the rotation
    :return: rotation matrix - 3x3 numpy array

    """
    if axis == 2:
        return np.array([[cosd(angle), -sind(angle), 0],
                         [sind(angle), cosd(angle), 0], [0, 0, 1]])
    elif axis == 1:
        return np.array([[cosd(angle), 0, -sind(angle)],
                         [0, 1, 0], [sind(angle), 0, cosd(angle)]])
    else:
        return np.array([[1, 0, 0], [0, cosd(angle), -sind(angle)],
                         [0, sind(angle), cosd(angle)]])


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


def cell2AB(cell):
    """Computes orthogonalization matrix from unit cell constants

    :param tuple cell: a,b,c, alpha, beta, gamma (degrees)
    :returns: tuple of two 3x3 numpy arrays (A,B)
        A for crystal(x) to Cartesian(X) transformations A*x = np.inner(A,x) =X
        B (= inverse of A) for Cartesian to crystal transformation
          B*X = np.inner(B,X) = x

        in reciprocal space
            X*  = B.T @ x*  or x @ B

        A = |ax  bx  cx|   B = |a*x a*y a*z|
            |ay  by  cy|       |b*x b*y b*z|
            |az  bz  cz|       |c*x c*y c*z|
    """
    G, g = cell2Gmat(cell)
    cellstar = Gmat2cell(G)
    A = np.zeros(shape=(3, 3))
    # from Giacovazzo (Fundamentals 2nd Ed.) p.75
    A[0, 0] = cell[0]                # a
    A[0, 1] = cell[1] * cosd(cell[5])  # b cos(gamma)
    A[0, 2] = cell[2] * cosd(cell[4])  # c cos(beta)
    A[1, 1] = cell[1] * sind(cell[5])  # b sin(gamma)
    # - c cos(alpha*) sin(beta)
    A[1, 2] = -cell[2] * cosd(cellstar[3]) * sind(cell[4])
    A[2, 2] = 1. / cellstar[2]         # 1/c*
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

#  #############################################


def twofold_reduce(twofold):
    def reduce(twofold):
        return {'sigma': twofold['sigma'][2:],
                'uvw': twofold['uvw'][2:], 'hkl': twofold['hkl'][2:],
                'dv': twofold['dv'][2:], 'rv': twofold['rv'][2:],
                'dbase': twofold['dbase'],
                'rbase': twofold['rbase'],
                'toll': twofold['toll']}
    out = [twofold]
    while True:
        if len(twofold['uvw']) == 1:
            break
        else:
            twofold = reduce(twofold)
            out.append(twofold)
    return out


def gen_twofoldmat():
    def evec(r):
        evalu, evec = nl.eig(r)
        u = evec.T[evalu.tolist().index(1)]
        u /= min([abs(i) for i in u if i != 0])
        return u.astype(int)
    z = []
    for i in nestedLoop(range(-1, 2), repeat=9):
        r = np.array(i).reshape(3, 3)
        if nl.det(r) != 1:
            continue
        if np.any(r @ r != np.identity(3)):
            continue
        if np.all(r == np.identity(3)):
            continue
        z.append({'r': r, 'uvw': evec(r), 'hkl': evec(r.T)})
    return z

all_twofold_ccbx = [
    {'r':(-1,-1,-1,0,0,1,0,1,0), 'uvw':(-1,1,1), 'hkl':(0,1,1)},
    {'r':(-1,-1,0,0,1,0,0,-1,-1), 'uvw':(1,-2,1), 'hkl':(0,1,0)},
    {'r':(-1,-1,0,0,1,0,0,0,-1), 'uvw':(-1,2,0), 'hkl':(0,1,0)},
    {'r':(-1,-1,0,0,1,0,0,1,-1), 'uvw':(-1,2,1), 'hkl':(0,1,0)},
    {'r':(-1,-1,1,0,0,-1,0,-1,0), 'uvw':(1,-1,1), 'hkl':(0,-1,1)},
    {'r':(-1,0,-1,0,-1,-1,0,0,1), 'uvw':(-1,-1,2), 'hkl':(0,0,1)},
    {'r':(-1,0,-1,0,-1,0,0,0,1), 'uvw':(-1,0,2), 'hkl':(0,0,1)},
    {'r':(-1,0,-1,0,-1,1,0,0,1), 'uvw':(-1,1,2), 'hkl':(0,0,1)},
    {'r':(-1,0,0,-1,0,-1,1,-1,0), 'uvw':(0,-1,1), 'hkl':(1,-1,1)},
    {'r':(-1,0,0,-1,0,1,-1,1,0), 'uvw':(0,1,1), 'hkl':(-1,1,1)},
    {'r':(-1,0,0,-1,1,-1,0,0,-1), 'uvw':(0,1,0), 'hkl':(1,-2,1)},
    {'r':(-1,0,0,-1,1,0,0,0,-1), 'uvw':(0,1,0), 'hkl':(-1,2,0)},
    {'r':(-1,0,0,-1,1,1,0,0,-1), 'uvw':(0,1,0), 'hkl':(-1,2,1)},
    {'r':(-1,0,0,0,-1,-1,0,0,1), 'uvw':(0,-1,2), 'hkl':(0,0,1)},
    {'r':(-1,0,0,0,-1,0,-1,-1,1), 'uvw':(0,0,1), 'hkl':(-1,-1,2)},
    {'r':(-1,0,0,0,-1,0,-1,0,1), 'uvw':(0,0,1), 'hkl':(-1,0,2)},
    {'r':(-1,0,0,0,-1,0,-1,1,1), 'uvw':(0,0,1), 'hkl':(-1,1,2)},
    {'r':(-1,0,0,0,-1,0,0,-1,1), 'uvw':(0,0,1), 'hkl':(0,-1,2)},
    {'r':(-1,0,0,0,-1,0,0,0,1), 'uvw':(0,0,1), 'hkl':(0,0,1)},
    {'r':(-1,0,0,0,-1,0,0,1,1), 'uvw':(0,0,1), 'hkl':(0,1,2)},
    {'r':(-1,0,0,0,-1,0,1,-1,1), 'uvw':(0,0,1), 'hkl':(1,-1,2)},
    {'r':(-1,0,0,0,-1,0,1,0,1), 'uvw':(0,0,1), 'hkl':(1,0,2)},
    {'r':(-1,0,0,0,-1,0,1,1,1), 'uvw':(0,0,1), 'hkl':(1,1,2)},
    {'r':(-1,0,0,0,-1,1,0,0,1), 'uvw':(0,1,2), 'hkl':(0,0,1)},
    {'r':(-1,0,0,0,0,-1,0,-1,0), 'uvw':(0,-1,1), 'hkl':(0,-1,1)},
    {'r':(-1,0,0,0,0,1,0,1,0), 'uvw':(0,1,1), 'hkl':(0,1,1)},
    {'r':(-1,0,0,0,1,-1,0,0,-1), 'uvw':(0,1,0), 'hkl':(0,-2,1)},
    {'r':(-1,0,0,0,1,0,0,-1,-1), 'uvw':(0,-2,1), 'hkl':(0,1,0)},
    {'r':(-1,0,0,0,1,0,0,0,-1), 'uvw':(0,1,0), 'hkl':(0,1,0)},
    {'r':(-1,0,0,0,1,0,0,1,-1), 'uvw':(0,2,1), 'hkl':(0,1,0)},
    {'r':(-1,0,0,0,1,1,0,0,-1), 'uvw':(0,1,0), 'hkl':(0,2,1)},
    {'r':(-1,0,0,1,0,-1,-1,-1,0), 'uvw':(0,-1,1), 'hkl':(-1,-1,1)},
    {'r':(-1,0,0,1,0,1,1,1,0), 'uvw':(0,1,1), 'hkl':(1,1,1)},
    {'r':(-1,0,0,1,1,-1,0,0,-1), 'uvw':(0,1,0), 'hkl':(-1,-2,1)},
    {'r':(-1,0,0,1,1,0,0,0,-1), 'uvw':(0,1,0), 'hkl':(1,2,0)},
    {'r':(-1,0,0,1,1,1,0,0,-1), 'uvw':(0,1,0), 'hkl':(1,2,1)},
    {'r':(-1,0,1,0,-1,-1,0,0,1), 'uvw':(1,-1,2), 'hkl':(0,0,1)},
    {'r':(-1,0,1,0,-1,0,0,0,1), 'uvw':(1,0,2), 'hkl':(0,0,1)},
    {'r':(-1,0,1,0,-1,1,0,0,1), 'uvw':(1,1,2), 'hkl':(0,0,1)},
    {'r':(-1,1,-1,0,0,-1,0,-1,0), 'uvw':(-1,-1,1), 'hkl':(0,-1,1)},
    {'r':(-1,1,0,0,1,0,0,-1,-1), 'uvw':(-1,-2,1), 'hkl':(0,1,0)},
    {'r':(-1,1,0,0,1,0,0,0,-1), 'uvw':(1,2,0), 'hkl':(0,1,0)},
    {'r':(-1,1,0,0,1,0,0,1,-1), 'uvw':(1,2,1), 'hkl':(0,1,0)},
    {'r':(-1,1,1,0,0,1,0,1,0), 'uvw':(1,1,1), 'hkl':(0,1,1)},
    {'r':(0,-1,-1,-1,0,1,0,0,-1), 'uvw':(-1,1,0), 'hkl':(-1,1,1)},
    {'r':(0,-1,-1,0,-1,0,-1,1,0), 'uvw':(-1,0,1), 'hkl':(-1,1,1)},
    {'r':(0,-1,0,-1,0,0,-1,1,-1), 'uvw':(-1,1,1), 'hkl':(-1,1,0)},
    {'r':(0,-1,0,-1,0,0,0,0,-1), 'uvw':(-1,1,0), 'hkl':(-1,1,0)},
    {'r':(0,-1,0,-1,0,0,1,-1,-1), 'uvw':(1,-1,1), 'hkl':(-1,1,0)},
    {'r':(0,-1,1,-1,0,-1,0,0,-1), 'uvw':(-1,1,0), 'hkl':(1,-1,1)},
    {'r':(0,-1,1,0,-1,0,1,-1,0), 'uvw':(1,0,1), 'hkl':(1,-1,1)},
    {'r':(0,0,-1,-1,-1,1,-1,0,0), 'uvw':(-1,1,1), 'hkl':(-1,0,1)},
    {'r':(0,0,-1,0,-1,0,-1,0,0), 'uvw':(-1,0,1), 'hkl':(-1,0,1)},
    {'r':(0,0,-1,1,-1,-1,-1,0,0), 'uvw':(-1,-1,1), 'hkl':(-1,0,1)},
    {'r':(0,0,1,-1,-1,-1,1,0,0), 'uvw':(1,-1,1), 'hkl':(1,0,1)},
    {'r':(0,0,1,0,-1,0,1,0,0), 'uvw':(1,0,1), 'hkl':(1,0,1)},
    {'r':(0,0,1,1,-1,1,1,0,0), 'uvw':(1,1,1), 'hkl':(1,0,1)},
    {'r':(0,1,-1,0,-1,0,-1,-1,0), 'uvw':(-1,0,1), 'hkl':(-1,-1,1)},
    {'r':(0,1,-1,1,0,-1,0,0,-1), 'uvw':(1,1,0), 'hkl':(-1,-1,1)},
    {'r':(0,1,0,1,0,0,-1,-1,-1), 'uvw':(-1,-1,1), 'hkl':(1,1,0)},
    {'r':(0,1,0,1,0,0,0,0,-1), 'uvw':(1,1,0), 'hkl':(1,1,0)},
    {'r':(0,1,0,1,0,0,1,1,-1), 'uvw':(1,1,1), 'hkl':(1,1,0)},
    {'r':(0,1,1,0,-1,0,1,1,0), 'uvw':(1,0,1), 'hkl':(1,1,1)},
    {'r':(0,1,1,1,0,1,0,0,-1), 'uvw':(1,1,0), 'hkl':(1,1,1)},
    {'r':(1,-1,-1,0,-1,0,0,0,-1), 'uvw':(1,0,0), 'hkl':(-2,1,1)},
    {'r':(1,-1,0,0,-1,0,0,0,-1), 'uvw':(1,0,0), 'hkl':(-2,1,0)},
    {'r':(1,-1,1,0,-1,0,0,0,-1), 'uvw':(1,0,0), 'hkl':(2,-1,1)},
    {'r':(1,0,-1,0,-1,0,0,0,-1), 'uvw':(1,0,0), 'hkl':(-2,0,1)},
    {'r':(1,0,0,-1,-1,0,-1,0,-1), 'uvw':(-2,1,1), 'hkl':(1,0,0)},
    {'r':(1,0,0,-1,-1,0,0,0,-1), 'uvw':(-2,1,0), 'hkl':(1,0,0)},
    {'r':(1,0,0,-1,-1,0,1,0,-1), 'uvw':(2,-1,1), 'hkl':(1,0,0)},
    {'r':(1,0,0,0,-1,0,-1,0,-1), 'uvw':(-2,0,1), 'hkl':(1,0,0)},
    {'r':(1,0,0,0,-1,0,0,0,-1), 'uvw':(1,0,0), 'hkl':(1,0,0)},
    {'r':(1,0,0,0,-1,0,1,0,-1), 'uvw':(2,0,1), 'hkl':(1,0,0)},
    {'r':(1,0,0,1,-1,0,-1,0,-1), 'uvw':(-2,-1,1), 'hkl':(1,0,0)},
    {'r':(1,0,0,1,-1,0,0,0,-1), 'uvw':(2,1,0), 'hkl':(1,0,0)},
    {'r':(1,0,0,1,-1,0,1,0,-1), 'uvw':(2,1,1), 'hkl':(1,0,0)},
    {'r':(1,0,1,0,-1,0,0,0,-1), 'uvw':(1,0,0), 'hkl':(2,0,1)},
    {'r':(1,1,-1,0,-1,0,0,0,-1), 'uvw':(1,0,0), 'hkl':(-2,-1,1)},
    {'r':(1,1,0,0,-1,0,0,0,-1), 'uvw':(1,0,0), 'hkl':(2,1,0)},
    {'r':(1,1,1,0,-1,0,0,0,-1), 'uvw':(1,0,0), 'hkl':(2,1,1)}]

all_twofold = [
    {'r': (-1, -1, -1, 0, 0, 1, 0, 1, 0), 'uvw': (-1, 1, 1), 'hkl': (0, 1, 1)},
    {'r': (-1, -1, 0, 0, 1, 0, 0, -1, -1), 'uvw': (-1, 2, -1), 'hkl': (0, 1, 0)},
    {'r': (-1, -1, 0, 0, 1, 0, 0, 0, -1), 'uvw': (-1, 2, 0), 'hkl': (0, 1, 0)},
    {'r': (-1, -1, 0, 0, 1, 0, 0, 1, -1), 'uvw': (-1, 2, 1), 'hkl': (0, 1, 0)},
    {'r': (-1, -1, 1, 0, 0, -1, 0, -1, 0), 'uvw': (-1, 1, -1), 'hkl': (0, -1, 1)},
    {'r': (-1, 0, -1, 0, -1, -1, 0, 0, 1), 'uvw': (-1, -1, 2), 'hkl': (0, 0, 1)},
    {'r': (-1, 0, -1, 0, -1, 0, 0, 0, 1), 'uvw': (-1, 0, 2), 'hkl': (0, 0, 1)},
    {'r': (-1, 0, -1, 0, -1, 1, 0, 0, 1), 'uvw': (-1, 1, 2), 'hkl': (0, 0, 1)},
    {'r': (-1, 0, 0, -1, 0, -1, 1, -1, 0), 'uvw': (0, -1, 1), 'hkl': (-1, 1, -1)},
    {'r': (-1, 0, 0, -1, 0, 1, -1, 1, 0), 'uvw': (0, 1, 1), 'hkl': (-1, 1, 1)},
    {'r': (-1, 0, 0, -1, 1, -1, 0, 0, -1), 'uvw': (0, 1, 0), 'hkl': (-1, 2, -1)},
    {'r': (-1, 0, 0, -1, 1, 0, 0, 0, -1), 'uvw': (0, 1, 0), 'hkl': (-1, 2, 0)},
    {'r': (-1, 0, 0, -1, 1, 1, 0, 0, -1), 'uvw': (0, 1, 0), 'hkl': (-1, 2, 1)},
    {'r': (-1, 0, 0, 0, -1, -1, 0, 0, 1), 'uvw': (0, -1, 2), 'hkl': (0, 0, 1)},
    {'r': (-1, 0, 0, 0, -1, 0, -1, -1, 1), 'uvw': (0, 0, 1), 'hkl': (-1, -1, 2)},
    {'r': (-1, 0, 0, 0, -1, 0, -1, 0, 1), 'uvw': (0, 0, 1), 'hkl': (-1, 0, 2)},
    {'r': (-1, 0, 0, 0, -1, 0, -1, 1, 1), 'uvw': (0, 0, 1), 'hkl': (-1, 1, 2)},
    {'r': (-1, 0, 0, 0, -1, 0, 0, -1, 1), 'uvw': (0, 0, 1), 'hkl': (0, -1, 2)},
    {'r': (-1, 0, 0, 0, -1, 0, 0, 0, 1), 'uvw': (0, 0, 1), 'hkl': (0, 0, 1)},
    {'r': (-1, 0, 0, 0, -1, 0, 0, 1, 1), 'uvw': (0, 0, 1), 'hkl': (0, 1, 2)},
    {'r': (-1, 0, 0, 0, -1, 0, 1, -1, 1), 'uvw': (0, 0, 1), 'hkl': (1, -1, 2)},
    {'r': (-1, 0, 0, 0, -1, 0, 1, 0, 1), 'uvw': (0, 0, 1), 'hkl': (1, 0, 2)},
    {'r': (-1, 0, 0, 0, -1, 0, 1, 1, 1), 'uvw': (0, 0, 1), 'hkl': (1, 1, 2)},
    {'r': (-1, 0, 0, 0, -1, 1, 0, 0, 1), 'uvw': (0, 1, 2), 'hkl': (0, 0, 1)},
    {'r': (-1, 0, 0, 0, 0, -1, 0, -1, 0), 'uvw': (0, -1, 1), 'hkl': (0, -1, 1)},
    {'r': (-1, 0, 0, 0, 0, 1, 0, 1, 0), 'uvw': (0, 1, 1), 'hkl': (0, 1, 1)},
    {'r': (-1, 0, 0, 0, 1, -1, 0, 0, -1), 'uvw': (0, 1, 0), 'hkl': (0, 2, -1)},
    {'r': (-1, 0, 0, 0, 1, 0, 0, -1, -1), 'uvw': (0, 2, -1), 'hkl': (0, 1, 0)},
    {'r': (-1, 0, 0, 0, 1, 0, 0, 0, -1), 'uvw': (0, 1, 0), 'hkl': (0, 1, 0)},
    {'r': (-1, 0, 0, 0, 1, 0, 0, 1, -1), 'uvw': (0, 2, 1), 'hkl': (0, 1, 0)},
    {'r': (-1, 0, 0, 0, 1, 1, 0, 0, -1), 'uvw': (0, 1, 0), 'hkl': (0, 2, 1)},
    {'r': (-1, 0, 0, 1, 0, -1, -1, -1, 0), 'uvw': (0, -1, 1), 'hkl': (1, 1, -1)},
    {'r': (-1, 0, 0, 1, 0, 1, 1, 1, 0), 'uvw': (0, 1, 1), 'hkl': (1, 1, 1)},
    {'r': (-1, 0, 0, 1, 1, -1, 0, 0, -1), 'uvw': (0, 1, 0), 'hkl': (1, 2, -1)},
    {'r': (-1, 0, 0, 1, 1, 0, 0, 0, -1), 'uvw': (0, 1, 0), 'hkl': (1, 2, 0)},
    {'r': (-1, 0, 0, 1, 1, 1, 0, 0, -1), 'uvw': (0, 1, 0), 'hkl': (1, 2, 1)},
    {'r': (-1, 0, 1, 0, -1, -1, 0, 0, 1), 'uvw': (1, -1, 2), 'hkl': (0, 0, 1)},
    {'r': (-1, 0, 1, 0, -1, 0, 0, 0, 1), 'uvw': (1, 0, 2), 'hkl': (0, 0, 1)},
    {'r': (-1, 0, 1, 0, -1, 1, 0, 0, 1), 'uvw': (1, 1, 2), 'hkl': (0, 0, 1)},
    {'r': (-1, 1, -1, 0, 0, -1, 0, -1, 0), 'uvw': (1, 1, -1), 'hkl': (0, -1, 1)},
    {'r': (-1, 1, 0, 0, 1, 0, 0, -1, -1), 'uvw': (1, 2, -1), 'hkl': (0, 1, 0)},
    {'r': (-1, 1, 0, 0, 1, 0, 0, 0, -1), 'uvw': (1, 2, 0), 'hkl': (0, 1, 0)},
    {'r': (-1, 1, 0, 0, 1, 0, 0, 1, -1), 'uvw': (1, 2, 1), 'hkl': (0, 1, 0)},
    {'r': (-1, 1, 1, 0, 0, 1, 0, 1, 0), 'uvw': (1, 1, 1), 'hkl': (0, 1, 1)},
    {'r': (0, -1, -1, -1, 0, 1, 0, 0, -1), 'uvw': (1, -1, 0), 'hkl': (-1, 1, 1)},
    {'r': (0, -1, -1, 0, -1, 0, -1, 1, 0), 'uvw': (1, 0, -1), 'hkl': (1, -1, -1)},
    {'r': (0, -1, 0, -1, 0, 0, -1, 1, -1), 'uvw': (-1, 1, 1), 'hkl': (1, -1, 0)},
    {'r': (0, -1, 0, -1, 0, 0, 0, 0, -1), 'uvw': (1, -1, 0), 'hkl': (1, -1, 0)},
    {'r': (0, -1, 0, -1, 0, 0, 1, -1, -1), 'uvw': (-1, 1, -1), 'hkl': (1, -1, 0)},
    {'r': (0, -1, 1, -1, 0, -1, 0, 0, -1), 'uvw': (1, -1, 0), 'hkl': (-1, 1, -1)},
    {'r': (0, -1, 1, 0, -1, 0, 1, -1, 0), 'uvw': (1, 0, 1), 'hkl': (1, -1, 1)},
    {'r': (0, 0, -1, -1, -1, 1, -1, 0, 0), 'uvw': (1, -1, -1), 'hkl': (1, 0, -1)},
    {'r': (0, 0, -1, 0, -1, 0, -1, 0, 0), 'uvw': (1, 0, -1), 'hkl': (1, 0, -1)},
    {'r': (0, 0, -1, 1, -1, -1, -1, 0, 0), 'uvw': (1, 1, -1), 'hkl': (1, 0, -1)},
    {'r': (0, 0, 1, -1, -1, -1, 1, 0, 0), 'uvw': (1, -1, 1), 'hkl': (1, 0, 1)},
    {'r': (0, 0, 1, 0, -1, 0, 1, 0, 0), 'uvw': (1, 0, 1), 'hkl': (1, 0, 1)},
    {'r': (0, 0, 1, 1, -1, 1, 1, 0, 0), 'uvw': (1, 1, 1), 'hkl': (1, 0, 1)},
    {'r': (0, 1, -1, 0, -1, 0, -1, -1, 0), 'uvw': (1, 0, -1), 'hkl': (1, 1, -1)},
    {'r': (0, 1, -1, 1, 0, -1, 0, 0, -1), 'uvw': (1, 1, 0), 'hkl': (1, 1, -1)},
    {'r': (0, 1, 0, 1, 0, 0, -1, -1, -1), 'uvw': (1, 1, -1), 'hkl': (1, 1, 0)},
    {'r': (0, 1, 0, 1, 0, 0, 0, 0, -1), 'uvw': (1, 1, 0), 'hkl': (1, 1, 0)},
    {'r': (0, 1, 0, 1, 0, 0, 1, 1, -1), 'uvw': (1, 1, 1), 'hkl': (1, 1, 0)},
    {'r': (0, 1, 1, 0, -1, 0, 1, 1, 0), 'uvw': (1, 0, 1), 'hkl': (1, 1, 1)},
    {'r': (0, 1, 1, 1, 0, 1, 0, 0, -1), 'uvw': (1, 1, 0), 'hkl': (1, 1, 1)},
    {'r': (1, -1, -1, 0, -1, 0, 0, 0, -1), 'uvw': (1, 0, 0), 'hkl': (2, -1, -1)},
    {'r': (1, -1, 0, 0, -1, 0, 0, 0, -1), 'uvw': (1, 0, 0), 'hkl': (2, -1, 0)},
    {'r': (1, -1, 1, 0, -1, 0, 0, 0, -1), 'uvw': (1, 0, 0), 'hkl': (2, -1, 1)},
    {'r': (1, 0, -1, 0, -1, 0, 0, 0, -1), 'uvw': (1, 0, 0), 'hkl': (2, 0, -1)},
    {'r': (1, 0, 0, -1, -1, 0, -1, 0, -1), 'uvw': (2, -1, -1), 'hkl': (1, 0, 0)},
    {'r': (1, 0, 0, -1, -1, 0, 0, 0, -1), 'uvw': (2, -1, 0), 'hkl': (1, 0, 0)},
    {'r': (1, 0, 0, -1, -1, 0, 1, 0, -1), 'uvw': (2, -1, 1), 'hkl': (1, 0, 0)},
    {'r': (1, 0, 0, 0, -1, 0, -1, 0, -1), 'uvw': (2, 0, -1), 'hkl': (1, 0, 0)},
    {'r': (1, 0, 0, 0, -1, 0, 0, 0, -1), 'uvw': (1, 0, 0), 'hkl': (1, 0, 0)},
    {'r': (1, 0, 0, 0, -1, 0, 1, 0, -1), 'uvw': (2, 0, 1), 'hkl': (1, 0, 0)},
    {'r': (1, 0, 0, 1, -1, 0, -1, 0, -1), 'uvw': (2, 1, -1), 'hkl': (1, 0, 0)},
    {'r': (1, 0, 0, 1, -1, 0, 0, 0, -1), 'uvw': (2, 1, 0), 'hkl': (1, 0, 0)},
    {'r': (1, 0, 0, 1, -1, 0, 1, 0, -1), 'uvw': (2, 1, 1), 'hkl': (1, 0, 0)},
    {'r': (1, 0, 1, 0, -1, 0, 0, 0, -1), 'uvw': (1, 0, 0), 'hkl': (2, 0, 1)},
    {'r': (1, 1, -1, 0, -1, 0, 0, 0, -1), 'uvw': (1, 0, 0), 'hkl': (2, 1, -1)},
    {'r': (1, 1, 0, 0, -1, 0, 0, 0, -1), 'uvw': (1, 0, 0), 'hkl': (2, 1, 0)},
    {'r': (1, 1, 1, 0, -1, 0, 0, 0, -1), 'uvw': (1, 0, 0), 'hkl': (2, 1, 1)}]

def search_twofold(cell, toll_angle):
    """
    base are columns vectors for the base
    based on:
    http://legacy.ccp4.ac.uk/newsletters/newsletter44/articles/explore_metric_symmetry.html
    cctbx_sources/cctbx/sgtbx/lattice_symmetry.cpp
    """
    toll = toll_angle * rpd

    if len(cell) == 6:
        d_base, r_base = cell2AB(cell)
    elif len(cell) == 3:
        d_base = cell
        r_base = nl.inv(cell)

    uvw, hkl, d_v, r_v, sigma = [], [], [], [], []

    for i in all_twofold:
        dv = np.dot(d_base, i['uvw'])
        rv = np.dot(i['hkl'], r_base)
        sig = mt.angle_between_vectors(dv, rv)
        if sig < toll:
            uvw.append(i['uvw'])
            hkl.append(i['hkl'])
            d_v.append(dv)
            r_v.append(rv)

            sigma.append(sig)
        order = np.flip(np.argsort(sigma))

    return {'sigma': np.degrees(np.array(sigma))[order],
            'uvw': np.array(uvw)[order], 'hkl': np.array(hkl)[order],
            'dv': np.array(d_v)[order], 'rv': np.array(r_v)[order],
            'dbase': d_base,
            'rbase': r_base,
            'toll': toll}


def get_cell(twofold):
    """dtwofold = uvw
       caxes = dv
       maxes = lenght dv
       rtwofold = hkl
       cross = sigma

       returns: 
                transformation matric for the cell
                new cell
        tr = |a1 a2 a3|
             |b1 b2 b3|
             |c1 c2 c3|

    new base = |a1x b1x c1x|
               |a1y b1y c1y| =  A @ tr.T
               |a1z b1z c1z|
    """
    def check90(angle):
        return abs(angle - (np.pi / 2)) < twofold['toll']

    def eqv(a, b, tola=0.2):
        return abs(a - b) < tola

    def colinear(v1, v2):
        return all(np.cross(v1, v2) == 0)

    tr = np.identity(3, dtype=int)
    d_base = twofold['dbase']
    ntwo = len(twofold['uvw'])

    if ntwo == 9:  # Case (9)   !Cubic n-2foldaxes=9
        mv = mt.mod(twofold['dv'])
        for i, j, k in combinations(range(ntwo), 3):
            vi = twofold['dv'][i]
            vj = twofold['dv'][j]
            vk = twofold['dv'][k]
            aij = mt.angle_between_vectors(vi, vj)
            aik = mt.angle_between_vectors(vi, vk)
            ajk = mt.angle_between_vectors(vj, vk)
            if check90(aij) and check90(aik) and check90(ajk):
                eij = eqv(mv[i], mv[j])
                eik = eqv(mv[i], mv[k])
                ejk = eqv(mv[j], mv[k])
                if eij and eik and ejk:
                    v1 = vi
                    v2 = vj
                    v3 = vk
                    tr = np.array([twofold['uvw'][i] for i in [i, j, k]])
                    break

        # check rightness
        namina = nl.det(tr)
        if namina < 0:
            tr[3] *= -1
            v3 *= -1
            namina *= -1

        if namina == 0:
            print("Pseudo-cubic but tolerance too small ... ")
            ok = False
            return
        if namina == 1:
            print("Cubic, Primitive cell")
        if namina == 2:
            if not(mt.coprime(np.dot([0, 1, 1], tr))):
                print("Cubic, A-centred cell")
            elif not(mt.coprime(np.dot([1, 1, 1], tr))):
                print("Cubic, I-centred cell")
            elif not(mt.coprime(np.dot([1, 1, 0], tr))):
                print("Cubic, C-centred cell")
            elif not(mt.coprime(np.dot([1, 0, 1], tr))):
                print("Cubic, B-centred cell")
        if namina >= 3:
            print("Cubic, F-centred cell")

    if ntwo == 7:  # Case (7)   !Hexagonal n-2foldaxes=7
        hexap = False
        hexac = False

        # Search tha a-b plane
        for i, j in combinations(range(ntwo), 2):
            vi = twofold['dv'][i]
            vj = twofold['dv'][j]
            aij = mt.angle_between_vectors(vi, vj)
            if abs(aij - (np.pi / 3.0)) < twofold['toll']:
                if (abs(mv[i] - mv[j]) < tola) and not(hexap):
                    v1 = vi
                    v2 = vj
                    hexap = True
                    break

        # then ! Search the c-axis, it should be also
        # a two-fold axis! because Op(6).Op(6).Op(6)=Op(2)
        if hexap:
            for i in range(ntwo):
                vi = twofold['dv'][i]
                aij = mt.angle_between_vectors(v1, vi)
                aik = mt.angle_between_vectors(v2, vi)
                if check90(aij) and check90(aik):
                    v3 = vi
                    hexac = True
                    break
            else:
                ok = False

        if hexac:
            tr = np.array([v1, v2, v3])
            namina = nl.det(tr)
            if (namina < 0):
                tr[3] *= -1
                v3 *= -1
                namina *= -1
            if namina == 1:
                print("Hexagonal, Primitive cell")
            if namina == 2:
                print("Hexagonal, centred cell? possible mistake")

        else:
            ok = False
            print("The c-axis of a hexagonal cell was not found!")
            return

    if ntwo == 5:  # Case (5)   !Tetragonal n-2foldaxes=5
        ab = []
        inp = np.zeros(5)
        mv = mt.mod(twofold['dv'])

        for i, j in combinations(range(ntwo), 2):
            vi = twofold['dv'][i]
            vj = twofold['dv'][j]
            m = mt.angle_between_vectors(vi, vj)
            c45 = abs(m - (np.pi * 0.25)) < twofold['toll']
            c135 = abs(m - (np.pi * 0.75)) < twofold['toll']
            if c45 or c135:
                inp[i] = 1
                inp[j] = 1
                ab.append([i, j][np.argmin(mv[[i, j]])])

        ab = list(set(ab))
        if len(ab) < 2:
            ok = False
            print("Basis vectors a-b not found!")
        #  !Determination of the c-axis
        #  (that making 90 degree with all the others)
        naminc = np.argmin(inp)
        #   !The two axes forming a,b are those of indices ab(1) and ab(2)
        namina = ab[0]
        naminb = ab[1]

        tr[0] = twofold['uvw'][namina]
        tr[1] = twofold['uvw'][naminb]
        tr[2] = twofold['uvw'][naminc]
        v1 = twofold['dv'][namina]
        v2 = twofold['dv'][naminb]
        v3 = twofold['dv'][naminc]

        namina = nl.det(tr)
        if (namina < 0):
            tr[2] = -tr[2]
            v3 = -v3
            namina = -namina

        if namina == 1:
            print("Tetragonal, Primitive cell")
        elif namina == 2:
            print("Tetragonal, I-centred cell")
        else:
            print("Error in tetragonal cell")
            ok = False

    if ntwo == 3:  # Case (3)   !Orthorhombic/Trigonal n-2foldaxes=3
        u = mt.mod(twofold['dv'])
        a_i = np.argmin(u)
        c_i = np.argmax(u)
        b_i = 3 - a_i - c_i
        tr[0, :] = twofold['uvw'][a_i]
        tr[1, :] = twofold['uvw'][b_i]
        tr[2, :] = twofold['uvw'][c_i]

        v1 = twofold['dv'][a_i]
        v2 = twofold['dv'][b_i]
        v3 = twofold['dv'][c_i]

        ang = np.array([mt.angle_between_vectors(v3, v2),
                        mt.angle_between_vectors(v1, v3),
                        mt.angle_between_vectors(v1, v3)])

        #  Check the system by verifying that the two-fold axes
        #  form 90 (orthorhombic)
        if all([check90(i) for i in ang]):  # if  !orthorhombic
            namina = nl.det(tr)
            if namina < 0:
                tr[2, :] *= -1
                v3 *= -1
                namina *= -1

            if namina == 1:
                print("Orthorhombic, Primitive cell")
            if namina == 2:
                vecs = [[0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1]]
                for rw_i, rw in enumerate(vecs):
                    if not(mt.coprime(np.dot(rw, tr))):
                        orthoType = rw_i
                        break
                message = ["Orthorhombic, A-centred cell",
                           "Orthorhombic, I-centred cell",
                           "Orthorhombic, C-centred cell",
                           "Orthorhombic, B-centred cell"]
                print(message[orthoType])
            if namina >= 3:
                print("Orthorhombic, F-centred cell")

        else:  # !Rhombohedral/Trigonal
            #  In the Trigonal system the two-fold axes are in the plane
            #  the three-fold axis, and valid a,b, vectors can be chosen
            #  among any two two-fold axes forming an angle of 120 degrees
            #  verify that 1 and 2 form 120          
            c_axis = None
            if any(abs(ang - (np.pi / 3)) < twofold['toll']):  # search 120
                c_axis = np.argmin(abs(ang - (np.pi / 6)))
                dot = 1.0
                iu = 1
            elif any(abs(ang - (np.pi / 6)) < twofold['toll']):  # search 60
                c_axis = np.armin(abs(ang - (np.pi / 6)))
                dot = -1.0
                iu = -1
            else:
                print("Trigonal/Rhombohedral test failed! \
                       Supply only one two-fold axis")
                return
            if c_axis == 0:
                vi = v2
                vj = dot * v3
                h1 = tr[1, :]
                h2 = iu * tr[2, :]
                tr[2, :] = tr[0, :]
                tr[0, :] = h1
                tr[1, :] = h2

            elif c_axis == 1:
                vi = v1
                vj = dot * v3
                h2 = iu * tr[2, :]
                tr[2, :] = tr[2, :]
                tr[1, :] = h2

            elif c_axis == 2:
                vi = v1
                vj = dot * v2
                tr[1, :] = iu * tr[1, :]

            ok = False

            for uvw in nestedLoop(range(-3, 4), range(-3, 4), range(0, 4)):
                if not(mt.coprime(uvw)):
                    continue
                vec = np.dot(d_base, uvw)
                ang1 = mt.angle_between_vectors(vec, vi)
                ang2 = mt.angle_between_vectors(vec, vj)
                if check90(ang1) and check90(ang2):
                    tr[2, :] = uvw
                    ok = True
                    break
            if ok:

                namina = np.round(nl.det(tr), 4)
                print(namina, namina == 3)
                if namina < 0:
                    tr[2, :] *= -1
                    namina *= -1
                v3 = np.dot(d_base, tr[2])
                if namina == 1:
                    print("Primitive hexagonal cell")

                elif namina == 3:
                    rw = np.dot([2, 1, 1], tr)
                    if not(mt.coprime(rw)):
                        print("Rhombohedral, obverse setting cell")
                    else:
                        print("Rhombohedral, reverse setting cell")
            else:
                print("Trigonal/Rhombohedral test failed!")
                print(" Supply only one two-fold axis")

    if ntwo == 1:  # Case (3)   !Monoclinic n-2foldaxes=1
        v2 = twofold['dv'][0]
        tr[1] = twofold['uvw'][0]
        row = []
        for rw in nestedLoop(range(-3, 4), range(-3, 4), range(0, 4)):
            if all(np.array(rw) == 0):
                continue
            if not(mt.coprime(rw)):
                continue
            vec = np.dot(d_base, rw)
            if check90(mt.angle_between_vectors(v2, vec)):
                if any([colinear(vec, i) for i in row]):
                    continue
                else:
                    row.append(np.array(rw))

        row_mod = np.array([mt.mod(np.dot(d_base, i)) for i in row])
        rms = np.argsort(row_mod)
        tr[0] = row[rms[0]]
        v1 = np.dot(d_base, tr[0, :])
        tr[2] = row[rms[1]]
        v3 = np.dot(d_base, tr[2, :])

        #  Test rightness
        if nl.det(tr) < 0:
            tr[1, :] = -tr[1, :]
            v2 = -v2

        # Test if beta is lower than 90 in such a case invert c and b
        if mt.angle_between_vectors(v1, v3) < (np.pi / 2):  # !angle beta < 90
            tr[0, :] = -tr[0, :]
            v1 = -v1
            tr[2, :] = -tr[2, :]
            v2 = -v2

        namina = nl.det(tr)
        if namina == 1:
            print("Monoclinic, Primitive cell")
        if namina == 2:
            if not(mt.coprime(np.dot([1, 1, 0], tr))):
                print("Monoclinic, C-centred cell")
            if not(mt.coprime(np.dot([0, 1, 1], tr))):
                print("Monoclinic, A-centred cell")

    if ntwo not in [1, 3, 5, 7, 9]:
        print("Wrong number of two-fold axes! ", ntwo)
        ok = False
        return
    # Calculation of the new cell
    alpha = np.degrees(mt.angle_between_vectors(v2, v3))
    beta = np.degrees(mt.angle_between_vectors(v1, v3))
    gamma = np.degrees(mt.angle_between_vectors(v1, v2))
    a = mt.mod(v1)
    b = mt.mod(v2)
    c = mt.mod(v3)
    print(f'a={round(a,2)}, b={round(b,2)}, c={round(c,2)},',
          f'alpha={round(alpha,2)}, beta={round(beta,2)},',
          f'gamma={round(gamma,2)}')
    ok = True
    return [a, b, c, alpha, beta, gamma], tr


def plot_twofold(twofold):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, two in enumerate(twofold['dv']):
        ax.plot([0, two[0]],
                [0, two[1]],
                [0, two[2]], label=str(i))
    plt.legend()

__test_cell = [{'prim': [4.0, 4.472, 4.583, 79.03, 64.13, 64.15], 'sigma': [1.48, 1.48, 0.714, 0.714, 0.005], 'sol': 'tetra I'},
               {'prim':[14.384, 14.38437, 13.439, 80.828, 99.172, 146.05], 'sigma': [0.0026], 'sol':'C2/m', 'id':'mp-570123'},
               {'prim':[6.1702, 6.17018, 6.1702, 121.48, 121.34, 87.572], 'sigma': [0.12, 0.12, 0.003, 0.003, 0.002], 'sol':'Imma', 'id':'mp-33684'},
               {'prim':[6.1702, 6.17018, 6.1702, 121.48, 121.34, 87.572], 'sigma': [0.12, 0.12, 0.003, 0.003, 0.002], 'sol':'Imma', 'id':'mp-33684'}]
