import numpy as np
import numpy.linalg as nl
import math_tools as mt

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
       A for crystal(x) to Cartesian(X) transformations A*x = np.inner(A,x) = X 
       B (= inverse of A) for Cartesian to crystal transformation B*X = np.inner(B,X) = x

       in reciprocal space 
       X*  = B.T @ x*  or x @ B 
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


all_twofold=[
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


def search_twofold(cell, toll_angle):
    """
    base are columns vectors for the base
    based on 
    http://legacy.ccp4.ac.uk/newsletters/newsletter44/articles/explore_metric_symmetry.html
    cctbx_sources/cctbx/sgtbx/lattice_symmetry.cpp
    """
    toll = toll_angle * rpd

    d_base, r_base = cell2AB(cell)


    uvw, hkl, d_v, r_v, sigma = [],[],[],[],[]
    caxes = []

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

    return {'uvw': np.array(uvw), 'hkl': np.array(hkl),
            'dv': np.array(d_v), 'rv': np.array(r_v),
            'sigma': np.degrees(np.array(sigma)),
            'dbase': d_base,
            'r_base': r_base,
            'toll': toll}


def get_cell(twofold, cell):
    tr = np.identity(3)
    ep = np.pi - twofold['toll']
    d_base = twofold['d_base']

    ntwo = len(twofold['uvw'])
    if ntwo == 1:              # !Monoclinic n-2foldaxes=1
        v2 = twofold['dv'][0]
        u = v2 / np.sqrt(v2 @ v2)
        tr[1, :] = twofold['uvw'][0]

        row = []
        zz = np.mgrid[-2:3, -2:3, 0:3]
        uvwl = zz.reshape(3, -1).T
        for rw in uvwl:
            if all(rw == 0):
                continue
            if np.reduce.gcd(rw) not in [0, 1]:
                continue
            vec = mt.norm(np.dot(d_base, rw))
            if (mt.angle_between_vectors(u, vec) > ep):
                if row:
                    if any([all(np.cross(rw, i) for i in row)]):
                        continue
                row.append(rw)

        row_mod = np.array([mt.mod(np.dot(d_base, i)) for i in row])
        rms = np.argsort(row_mod)
        tr[0, :] = row[rms[0]]
        v1 = np.dot(d_base, tr[0, :])
        tr[2, :] = row[rms[1]]
        v3 = np.dot(d_base, tr[2, :])

        #  Test rightness
        if nl.det(tr) < 0:
            tr[1, :] = -tr[1, :]
            v2 = -v2


        #Test if beta is lower than 90 in such a case invert c and b
        if mt.angle_between_vectors(v1, v3) < (np.pi / 2):  # !angle beta < 90
            tr[0, :] = -tr[0, :]
            v1 = -v1
            tr[2, :] = -tr[2, :]
            v2 = -v2
