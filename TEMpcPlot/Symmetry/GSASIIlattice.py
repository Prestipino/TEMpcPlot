# -*- coding: utf-8 -*-
'''
*GSASIIlattice: Unit cells*
---------------------------

Perform lattice-related computations

Note that *G* is the reciprocal lattice tensor, and *g* is its inverse,
:math:`G = g^{-1}`, where 

  .. math::

   g = \\left( \\begin{matrix}
   a^2 & a b\\cos\gamma & a c\\cos\\beta \\\\
   a b\\cos\\gamma & b^2 & b c \cos\\alpha \\\\
   a c\\cos\\beta &  b c \\cos\\alpha & c^2
   \\end{matrix}\\right)

The "*A* tensor" terms are defined as
:math:`A = (\\begin{matrix} G_{11} & G_{22} & G_{33} & 2G_{12} & 2G_{13} & 2G_{23}\\end{matrix})` and *A* can be used in this fashion:
:math:`d^* = \sqrt {A_1 h^2 + A_2 k^2 + A_3 l^2 + A_4 hk + A_5 hl + A_6 kl}`, where
*d* is the d-spacing, and :math:`d^*` is the reciprocal lattice spacing, 
:math:`Q = 2 \\pi d^* = 2 \\pi / d`
'''
########### SVN repository information ###################
# $Date$
# $Author$
# $Revision$
# $URL$
# $Id$
########### SVN repository information ###################
from __future__ import division, print_function
import math
import copy
import sys
import random as ran
import numpy as np
import numpy.linalg as nl



# trig functions in degrees
sind = lambda x: np.sin(x*np.pi/180.)
asind = lambda x: 180.*np.arcsin(x)/np.pi
tand = lambda x: np.tan(x*np.pi/180.)
atand = lambda x: 180.*np.arctan(x)/np.pi
atan2d = lambda y,x: 180.*np.arctan2(y,x)/np.pi
cosd = lambda x: np.cos(x*np.pi/180.)
acosd = lambda x: 180.*np.arccos(x)/np.pi
rdsq2d = lambda x,p: round(1.0/np.sqrt(x),p)
rpd = np.pi/180.
RSQ2PI = 1./np.sqrt(2.*np.pi)
SQ2 = np.sqrt(2.)
RSQPI = 1./np.sqrt(np.pi)
R2pisq = 1./(2.*np.pi**2)
nxs = np.newaxis

def sec2HMS(sec):
    """Convert time in sec to H:M:S string
    
    :param sec: time in seconds
    :return: H:M:S string (to nearest 100th second)
    
    """
    H = int(sec//3600)
    M = int(sec//60-H*60)
    S = sec-3600*H-60*M
    return '%d:%2d:%.2f'%(H,M,S)
    
def rotdMat(angle,axis=0):
    """Prepare rotation matrix for angle in degrees about axis(=0,1,2)

    :param angle: angle in degrees
    :param axis:  axis (0,1,2 = x,y,z) about which for the rotation
    :return: rotation matrix - 3x3 numpy array

    """
    if axis == 2:
        return np.array([[cosd(angle),-sind(angle),0],[sind(angle),cosd(angle),0],[0,0,1]])
    elif axis == 1:
        return np.array([[cosd(angle),0,-sind(angle)],[0,1,0],[sind(angle),0,cosd(angle)]])
    else:
        return np.array([[1,0,0],[0,cosd(angle),-sind(angle)],[0,sind(angle),cosd(angle)]])
        
def rotdMat4(angle,axis=0):
    """Prepare rotation matrix for angle in degrees about axis(=0,1,2) with scaling for OpenGL 

    :param angle: angle in degrees
    :param axis:  axis (0,1,2 = x,y,z) about which for the rotation
    :return: rotation matrix - 4x4 numpy array (last row/column for openGL scaling)

    """
    Mat = rotdMat(angle,axis)
    return np.concatenate((np.concatenate((Mat,[[0],[0],[0]]),axis=1),[[0,0,0,1],]),axis=0)
    
def fillgmat(cell):
    """Compute lattice metric tensor from unit cell constants

    :param cell: tuple with a,b,c,alpha, beta, gamma (degrees)
    :return: 3x3 numpy array

    """
    a,b,c,alp,bet,gam = cell
    g = np.array([
        [a*a,  a*b*cosd(gam),  a*c*cosd(bet)],
        [a*b*cosd(gam),  b*b,  b*c*cosd(alp)],
        [a*c*cosd(bet) ,b*c*cosd(alp),   c*c]])
    return g
           
def cell2Gmat(cell):
    """Compute real and reciprocal lattice metric tensor from unit cell constants

    :param cell: tuple with a,b,c,alpha, beta, gamma (degrees)
    :return: reciprocal (G) & real (g) metric tensors (list of two numpy 3x3 arrays)

    """
    g = fillgmat(cell)
    G = nl.inv(g)        
    return G,g

def A2Gmat(A,inverse=True):
    """Fill real & reciprocal metric tensor (G) from A.

    :param A: reciprocal metric tensor elements as [G11,G22,G33,2*G12,2*G13,2*G23]
    :param bool inverse: if True return both G and g; else just G
    :return: reciprocal (G) & real (g) metric tensors (list of two numpy 3x3 arrays)

    """
    G = np.array([
        [A[0],  A[3]/2.,  A[4]/2.], 
        [A[3]/2.,A[1],    A[5]/2.], 
        [A[4]/2.,A[5]/2.,    A[2]]])
    if inverse:
        g = nl.inv(G)
        return G,g
    else:
        return G

def Gmat2A(G):
    """Extract A from reciprocal metric tensor (G)

    :param G: reciprocal maetric tensor (3x3 numpy array
    :return: A = [G11,G22,G33,2*G12,2*G13,2*G23]

    """
    return [G[0][0],G[1][1],G[2][2],2.*G[0][1],2.*G[0][2],2.*G[1][2]]
    
def cell2A(cell):
    """Obtain A = [G11,G22,G33,2*G12,2*G13,2*G23] from lattice parameters

    :param cell: [a,b,c,alpha,beta,gamma] (degrees)
    :return: G reciprocal metric tensor as 3x3 numpy array

    """
    G,g = cell2Gmat(cell)
    return Gmat2A(G)

def A2cell(A):
    """Compute unit cell constants from A

    :param A: [G11,G22,G33,2*G12,2*G13,2*G23] G - reciprocal metric tensor
    :return: a,b,c,alpha, beta, gamma (degrees) - lattice parameters

    """
    G,g = A2Gmat(A)
    return Gmat2cell(g)

def Gmat2cell(g):
    """Compute real/reciprocal lattice parameters from real/reciprocal metric tensor (g/G)
    The math works the same either way.

    :param g (or G): real (or reciprocal) metric tensor 3x3 array
    :return: a,b,c,alpha, beta, gamma (degrees) (or a*,b*,c*,alpha*,beta*,gamma* degrees)

    """
    oldset = np.seterr('raise')
    a = np.sqrt(max(0,g[0][0]))
    b = np.sqrt(max(0,g[1][1]))
    c = np.sqrt(max(0,g[2][2]))
    alp = acosd(g[2][1]/(b*c))
    bet = acosd(g[2][0]/(a*c))
    gam = acosd(g[0][1]/(a*b))
    np.seterr(**oldset)
    return a,b,c,alp,bet,gam

def invcell2Gmat(invcell):
    """Compute real and reciprocal lattice metric tensor from reciprocal 
       unit cell constants
       
    :param invcell: [a*,b*,c*,alpha*, beta*, gamma*] (degrees)
    :return: reciprocal (G) & real (g) metric tensors (list of two 3x3 arrays)

    """
    G = fillgmat(invcell)
    g = nl.inv(G)
    return G, g


    
def prodMGMT(G,Mat):
    '''Transform metric tensor by matrix
    
    :param G: array metric tensor
    :param Mat: array transformation matrix
    :return: array new metric tensor
    
    '''
    return np.inner(np.inner(Mat,G),Mat)        #right
#    return np.inner(Mat,np.inner(Mat,G))       #right
#    return np.inner(np.inner(G,Mat).T,Mat)      #right
#    return np.inner(Mat,np.inner(G,Mat).T)     #right
    
def TransformCell(cell,Trans):
    '''Transform lattice parameters by matrix
    
    :param cell: list a,b,c,alpha,beta,gamma,(volume)
    :param Trans: array transformation matrix
    :return: array transformed a,b,c,alpha,beta,gamma,volume
    
    '''
    newCell = np.zeros(7)
    g = cell2Gmat(cell)[1]
    newg = prodMGMT(g,Trans)
    newCell[:6] = Gmat2cell(newg)
    newCell[6] = calc_V(cell2A(newCell[:6]))
    return newCell
    
def TransformXYZ(XYZ, Trans, Vec):
    return np.inner(XYZ,Trans)+Vec
    
def TransformU6(U6, Trans):
    Uij = np.inner(Trans,np.inner(U6toUij(U6),Trans).T)/nl.det(Trans)
    return UijtoU6(Uij)

def ExpandCell(Atoms,atCodes,cx,Trans):
    Unit =[int(max(abs(np.array(unit)))-1) for unit in Trans.T]
    for i,unit in enumerate(Unit):
        if unit > 0:
            for j in range(unit):
                moreAtoms = copy.deepcopy(Atoms)
                moreCodes = []
                for atom,code in zip(moreAtoms,atCodes):
                    atom[cx+i] += 1.
                    if '+' in code:
                        cell = list(eval(code.split('+')[1]))
                        ops = code.split('+')[0]
                    else:
                        cell = [0,0,0]
                        ops = code
                    cell[i] += 1
                    moreCodes.append('%s+%d,%d,%d'%(ops,cell[0],cell[1],cell[2])) 
                Atoms += moreAtoms
                atCodes += moreCodes
    return Atoms,atCodes
    
def TransformPhase(oldPhase, newPhase,Trans,Uvec,Vvec,ifMag):
    '''Transform atoms from oldPhase to newPhase
    M' is inv(M)
    does X' = M(X-U)+V transformation for coordinates and U' = MUM/det(M)
    for anisotropic thermal parameters
    
    :param oldPhase: dict G2 phase info for old phase
    :param newPhase: dict G2 phase info for new phase; with new cell & space group
            atoms are from oldPhase & will be transformed
    :param Trans: lattice transformation matrix M
    :param Uvec: array parent coordinates transformation vector U
    :param Vvec: array child coordinate transformation vector V
    :param ifMag: bool True if convert to magnetic phase; 
        if True all nonmagnetic atoms will be removed
        
    :return: newPhase dict modified G2 phase info
    :return: atCodes list atom transformation codes
        
    '''
    
    cx,ct,cs,cia = oldPhase['General']['AtomPtrs']
    cm = 0
    if oldPhase['General']['Type'] == 'magnetic':
        cm = cx+4
    oAmat,oBmat = cell2AB(oldPhase['General']['Cell'][1:7])
    nAmat,nBmat = cell2AB(newPhase['General']['Cell'][1:7])
    SGData = newPhase['General']['SGData']
    invTrans = nl.inv(Trans)
    newAtoms,atCodes = FillUnitCell(oldPhase)
    newAtoms,atCodes = ExpandCell(newAtoms,atCodes,cx,Trans)
    if ifMag:
        cia += 3
        cs += 3
        newPhase['General']['Type'] = 'magnetic'
        newPhase['General']['AtomPtrs'] = [cx,ct,cs,cia]
        magAtoms = []
        magatCodes = []
        Landeg = 2.0
        for iat,atom in enumerate(newAtoms):
            if len(G2elem.GetMFtable([atom[ct],],[Landeg,])):
                magAtoms.append(atom[:cx+4]+[0.,0.,0.]+atom[cx+4:])
                magatCodes.append(atCodes[iat])
        newAtoms = magAtoms
        atCodes = magatCodes
        newPhase['Draw Atoms'] = []
    for atom in newAtoms:
        xyz = TransformXYZ(atom[cx:cx+3]+Uvec,invTrans.T,Vvec)%1.
        atom[cx:cx+3] = np.around(xyz,6)%1.
        if atom[cia] == 'A':
            atom[cia+2:cia+8] = TransformU6(atom[cia+2:cia+8],Trans)
        atom[cs:cs+2] = G2spc.SytSym(atom[cx:cx+3],SGData)[:2]
        atom[cia+8] = ran.randint(0,sys.maxsize)
        if cm:
            mag = np.sqrt(np.sum(np.array(atom[cm:cm+3])**2))
            if mag:
                mom = np.inner(np.array(atom[cm:cm+3]),oBmat)
                mom = np.inner(mom,invTrans)
                mom = np.inner(mom,nAmat)
                mom /= np.sqrt(np.sum(mom**2))
                atom[cm:cm+3] = mom*mag
    newPhase['Atoms'] = newAtoms
    newPhase['Atoms'],atCodes = GetUnique(newPhase,atCodes)
    newPhase['Drawing'] = []
    newPhase['ranId'] = ran.randint(0,sys.maxsize)
    return newPhase,atCodes
    

            
def calc_rVsq(A):
    """Compute the square of the reciprocal lattice volume (1/V**2) from A'

    """
    G,g = A2Gmat(A)
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
    return 1./calc_rV(A)

def A2invcell(A):
    """Compute reciprocal unit cell constants from A
    returns tuple with a*,b*,c*,alpha*, beta*, gamma* (degrees)
    """
    G,g = A2Gmat(A)
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
    G,g = cell2Gmat(cell) 
    cellstar = Gmat2cell(G)
    A = np.zeros(shape=(3,3))
    # from Giacovazzo (Fundamentals 2nd Ed.) p.75
    A[0][0] = cell[0]                # a
    A[0][1] = cell[1]*cosd(cell[5])  # b cos(gamma)
    A[0][2] = cell[2]*cosd(cell[4])  # c cos(beta)
    A[1][1] = cell[1]*sind(cell[5])  # b sin(gamma)
    A[1][2] = -cell[2]*cosd(cellstar[3])*sind(cell[4]) # - c cos(alpha*) sin(beta)
    A[2][2] = 1./cellstar[2]         # 1/c*
    B = nl.inv(A)
    return A,B
    
def HKL2SpAng(H,cell,SGData):
    """Computes spherical coords for hkls; view along 001

    :param array H: arrays of hkl
    :param tuple cell: a,b,c, alpha, beta, gamma (degrees)
    :param dict SGData: space group dictionary
    :returns: arrays of r,phi,psi (radius,inclination,azimuth) about 001 
    """
    A,B = cell2AB(cell)
    xH = np.inner(B.T,H)
    r = np.sqrt(np.sum(xH**2,axis=0))
    phi = acosd(xH[2]/r)
    psi = atan2d(xH[1],xH[0])
    phi = np.where(phi>90.,180.-phi,phi)
#    GSASIIpath.IPyBreak()
    return r,phi,psi
    
def U6toUij(U6):
    """Fill matrix (Uij) from U6 = [U11,U22,U33,U12,U13,U23]
    NB: there is a non numpy version in GSASIIspc: U2Uij

    :param list U6: 6 terms of u11,u22,...
    :returns:
        Uij - numpy [3][3] array of uij
    """
    U = np.array([
        [U6[0],  U6[3],  U6[4]], 
        [U6[3],  U6[1],  U6[5]], 
        [U6[4],  U6[5],  U6[2]]])
    return U

def UijtoU6(U):
    """Fill vector [U11,U22,U33,U12,U13,U23] from Uij 
    NB: there is a non numpy version in GSASIIspc: Uij2U
    """
    U6 = np.array([U[0][0],U[1][1],U[2][2],U[0][1],U[0][2],U[1][2]])
    return U6

def betaij2Uij(betaij,G):
    """
    Convert beta-ij to Uij tensors
    
    :param beta-ij - numpy array [beta-ij]
    :param G: reciprocal metric tensor
    :returns: Uij: numpy array [Uij]
    """
    ast = np.sqrt(np.diag(G))   #a*, b*, c*
    Mast = np.multiply.outer(ast,ast)    
    return R2pisq*UijtoU6(U6toUij(betaij)/Mast)
    
def Uij2betaij(Uij,G):
    """
    Convert Uij to beta-ij tensors -- stub for eventual completion
    
    :param Uij: numpy array [Uij]
    :param G: reciprocal metric tensor
    :returns: beta-ij - numpy array [beta-ij]
    """
    pass
    
def cell2GS(cell):
    ''' returns Uij to betaij conversion matrix'''
    G,g = cell2Gmat(cell)
    GS = G
    GS[0][1] = GS[1][0] = math.sqrt(GS[0][0]*GS[1][1])
    GS[0][2] = GS[2][0] = math.sqrt(GS[0][0]*GS[2][2])
    GS[1][2] = GS[2][1] = math.sqrt(GS[1][1]*GS[2][2])
    return GS    
    
def Uij2Ueqv(Uij,GS,Amat):
    ''' returns 1/3 trace of diagonalized U matrix'''
    U = np.multiply(U6toUij(Uij),GS)
    U = np.inner(Amat,np.inner(U,Amat).T)
    E,R = nl.eigh(U)
    return np.sum(E)/3.
        
def CosAngle(U,V,G):
    """ calculate cos of angle between U & V in generalized coordinates 
    defined by metric tensor G

    :param U: 3-vectors assume numpy arrays, can be multiple reflections as (N,3) array
    :param V: 3-vectors assume numpy arrays, only as (3) vector
    :param G: metric tensor for U & V defined space assume numpy array
    :returns:
        cos(phi)
    """
    u = (U.T/np.sqrt(np.sum(np.inner(U,G)*U,axis=1))).T
    v = V/np.sqrt(np.inner(V,np.inner(G,V)))
    cosP = np.inner(u,np.inner(G,v))
    return cosP
    
def CosSinAngle(U,V,G):
    """ calculate sin & cos of angle between U & V in generalized coordinates 
    defined by metric tensor G

    :param U: 3-vectors assume numpy arrays
    :param V: 3-vectors assume numpy arrays
    :param G: metric tensor for U & V defined space assume numpy array
    :returns:
        cos(phi) & sin(phi)
    """
    u = U/np.sqrt(np.inner(U,np.inner(G,U)))
    v = V/np.sqrt(np.inner(V,np.inner(G,V)))
    cosP = np.inner(u,np.inner(G,v))
    sinP = np.sqrt(max(0.0,1.0-cosP**2))
    return cosP,sinP
    
def criticalEllipse(prob):
    """
    Calculate critical values for probability ellipsoids from probability
    """
    if not ( 0.01 <= prob < 1.0):
        return 1.54 
    coeff = np.array([6.44988E-09,4.16479E-07,1.11172E-05,1.58767E-04,0.00130554,
        0.00604091,0.0114921,-0.040301,-0.6337203,1.311582])
    llpr = math.log(-math.log(prob))
    return np.polyval(coeff,llpr)
    
def CellBlock(nCells):
    """
    Generate block of unit cells n*n*n on a side; [0,0,0] centered, n = 2*nCells+1
    currently only works for nCells = 0 or 1 (not >1)
    """
    if nCells:
        N = 2*nCells+1
        N2 = N*N
        N3 = N*N*N
        cellArray = []
        A = np.array(range(N3))
        cellGen = np.array([A//N2-1,A//N%N-1,A%N-1]).T
        for cell in cellGen:
            cellArray.append(cell)
        return cellArray
    else:
        return [0,0,0]
        
def CellAbsorption(ElList,Volume):
    '''Compute unit cell absorption

    :param dict ElList: dictionary of element contents including mu and
      number of atoms be cell
    :param float Volume: unit cell volume
    :returns: mu-total/Volume
    '''
    muT = 0
    for El in ElList:
        muT += ElList[El]['mu']*ElList[El]['FormulaNo']
    return muT/Volume
    
#Permutations and Combinations
# Four routines: combinations,uniqueCombinations, selections & permutations
#These taken from Python Cookbook, 2nd Edition. 19.15 p724-726
#    
def _combinators(_handle, items, n):
    """ factored-out common structure of all following combinators """
    if n==0:
        yield [ ]
        return
    for i, item in enumerate(items):
        this_one = [ item ]
        for cc in _combinators(_handle, _handle(items, i), n-1):
            yield this_one + cc
def combinations(items, n):
    """ take n distinct items, order matters """
    def skipIthItem(items, i):
        return items[:i] + items[i+1:]
    return _combinators(skipIthItem, items, n)
def uniqueCombinations(items, n):
    """ take n distinct items, order is irrelevant """
    def afterIthItem(items, i):
        return items[i+1:]
    return _combinators(afterIthItem, items, n)
def selections(items, n):
    """ take n (not necessarily distinct) items, order matters """
    def keepAllItems(items, i):
        return items
    return _combinators(keepAllItems, items, n)
def permutations(items):
    """ take all items, order matters """
    return combinations(items, len(items))

#reflection generation routines
#for these: H = [h,k,l]; A is as used in calc_rDsq; G - inv metric tensor, g - metric tensor; 
#           cell - a,b,c,alp,bet,gam in A & deg
   
def Pos2dsp(Inst,pos):
    ''' convert powder pattern position (2-theta or TOF, musec) to d-spacing
    '''
    if 'C' in Inst['Type'][0] or 'PKS' in Inst['Type'][0]:
        wave = G2mth.getWave(Inst)
        return wave/(2.0*sind((pos-Inst.get('Zero',[0,0])[1])/2.0))
    else:   #'T'OF - ignore difB
        return TOF2dsp(Inst,pos)
        
def TOF2dsp(Inst,Pos):
    ''' convert powder pattern TOF, musec to d-spacing by successive approximation
    Pos can be numpy array
    '''
    def func(d,pos,Inst):        
        return (pos-Inst['difA'][1]*d**2-Inst['Zero'][1]-Inst['difB'][1]/d)/Inst['difC'][1]
    dsp0 = np.ones_like(Pos)
    N = 0
    while True:      #successive approximations
        dsp = func(dsp0,Pos,Inst)
        if np.allclose(dsp,dsp0,atol=0.000001):
            return dsp
        dsp0 = dsp
        N += 1
        if N > 10:
            return dsp
    
def Dsp2pos(Inst,dsp):
    ''' convert d-spacing to powder pattern position (2-theta or TOF, musec)
    '''
    if 'C' in Inst['Type'][0] or 'PKS' in Inst['Type'][0]:
        wave = G2mth.getWave(Inst)
        val = min(0.995,wave/(2.*dsp))  #set max at 168deg
        pos = 2.0*asind(val)+Inst.get('Zero',[0,0])[1]             
    else:   #'T'OF
        pos = Inst['difC'][1]*dsp+Inst['Zero'][1]+Inst['difA'][1]*dsp**2+Inst.get('difB',[0,0,False])[1]/dsp
    return pos
    
def getPeakPos(dataType,parmdict,dsp):
    ''' convert d-spacing to powder pattern position (2-theta or TOF, musec)
    '''
    if 'C' in dataType:
        pos = 2.0*asind(parmdict['Lam']/(2.*dsp))+parmdict['Zero']
    else:   #'T'OF
        pos = parmdict['difC']*dsp+parmdict['difA']*dsp**2+parmdict['difB']/dsp+parmdict['Zero']
    return pos
                   
def calc_rDsq(H,A):
    'needs doc string'
    rdsq = H[0]*H[0]*A[0]+H[1]*H[1]*A[1]+H[2]*H[2]*A[2]+H[0]*H[1]*A[3]+H[0]*H[2]*A[4]+H[1]*H[2]*A[5]
    return rdsq
    
def calc_rDsq2(H,G):
    'needs doc string'
    return np.inner(H,np.inner(G,H))
    
def calc_rDsqSS(H,A,vec):
    'needs doc string'
    rdsq = calc_rDsq(H[:3]+(H[3]*vec).T,A)
    return rdsq
       
def calc_rDsqZ(H,A,Z,tth,lam):
    'needs doc string'
    rdsq = calc_rDsq(H,A)+Z*sind(tth)*2.0*rpd/lam**2
    return rdsq
       
def calc_rDsqZSS(H,A,vec,Z,tth,lam):
    'needs doc string'
    rdsq = calc_rDsq(H[:3]+(H[3][:,np.newaxis]*vec).T,A)+Z*sind(tth)*2.0*rpd/lam**2
    return rdsq
       
def calc_rDsqT(H,A,Z,tof,difC):
    'needs doc string'
    rdsq = calc_rDsq(H,A)+Z/difC
    return rdsq
       
def calc_rDsqTSS(H,A,vec,Z,tof,difC):
    'needs doc string'
    rdsq = calc_rDsq(H[:3]+(H[3][:,np.newaxis]*vec).T,A)+Z/difC
    return rdsq
    
def PlaneIntercepts(Amat,H,phase,stack):
    ''' find unit cell intercepts for a stack of hkl planes
    '''
    Steps = range(-1,2,2)
    if stack:
        Steps = range(-10,10,1)
    Stack = []
    Ux = np.array([[0,0],[1,0],[1,1],[0,1]])
    for step in Steps:
        HX = []
        for i in [0,1,2]:
            if H[i]:
               h,k,l = [(i+1)%3,(i+2)%3,(i+3)%3]
               for j in [0,1,2,3]:
                    hx = [0,0,0]
                    intcpt = ((phase)/360.+step-H[h]*Ux[j,0]-H[k]*Ux[j,1])/H[l]
                    if 0. <= intcpt <= 1.:                        
                        hx[h] = Ux[j,0]
                        hx[k] = Ux[j,1]
                        hx[l] = intcpt
                        HX.append(hx)
        if len(HX)> 2:
            HX = np.array(HX)
            DX = np.inner(HX-HX[0],Amat)
            D = np.sqrt(np.sum(DX**2,axis=1))
            Dsort = np.argsort(D)
            HX = HX[Dsort]
            DX = DX[Dsort]
            D = D[Dsort]
            DX[1:,:] = DX[1:,:]/D[1:,nxs]
            A = 2.*np.ones(HX.shape[0])
            A[1:] = [np.dot(DX[1],dx) for dx in DX[1:]]
            HX = HX[np.argsort(A)]
            Stack.append(HX)
    return Stack
       
def MaxIndex(dmin,A):
    'needs doc string'
    Hmax = [0,0,0]
    try:
        cell = A2cell(A)
    except:
        cell = [1.,1.,1.,90.,90.,90.]
    for i in range(3):
        Hmax[i] = int(round(cell[i]/dmin))
    return Hmax
    
def transposeHKLF(transMat,Super,refList):
    ''' Apply transformation matrix to hkl(m)
    param: transmat: 3x3 or 4x4 array
    param: Super: 0 or 1 for extra index
    param: refList list of h,k,l,....
    return: newRefs transformed list of h',k',l',,,
    return: badRefs list of noninteger h',k',l'...
    '''
    newRefs = np.copy(refList)
    badRefs = []
    for H in newRefs:
        newH = np.inner(transMat,H[:3+Super])
        H[:3+Super] = np.rint(newH)
        if not np.allclose(newH,H[:3+Super],atol=0.01):
            badRefs.append(newH)
    return newRefs,badRefs
    
def sortHKLd(HKLd,ifreverse,ifdup,ifSS=False):
    '''sort reflection list on d-spacing; can sort in either order

    :param HKLd: a list of [h,k,l,d,...];
    :param ifreverse: True for largest d first
    :param ifdup: True if duplicate d-spacings allowed
    :return: sorted reflection list
    '''
    T = []
    N = 3
    if ifSS:
        N = 4
    for i,H in enumerate(HKLd):
        if ifdup:
            T.append((H[N],i))
        else:
            T.append(H[N])            
    D = dict(zip(T,HKLd))
    T.sort()
    if ifreverse:
        T.reverse()
    X = []
    okey = ''
    for key in T: 
        if key != okey: X.append(D[key])    #remove duplicate d-spacings
        okey = key
    return X
    
def SwapIndx(Axis,H):
    'needs doc string'
    if Axis in [1,-1]:
        return H
    elif Axis in [2,-3]:
        return [H[1],H[2],H[0]]
    else:
        return [H[2],H[0],H[1]]
        
def Rh2Hx(Rh):
    'needs doc string'
    Hx = [0,0,0]
    Hx[0] = Rh[0]-Rh[1]
    Hx[1] = Rh[1]-Rh[2]
    Hx[2] = np.sum(Rh)
    return Hx
    
def Hx2Rh(Hx):
    'needs doc string'
    Rh = [0,0,0]
    itk = -Hx[0]+Hx[1]+Hx[2]
    if itk%3 != 0:
        return 0        #error - not rhombohedral reflection
    else:
        Rh[1] = itk//3
        Rh[0] = Rh[1]+Hx[0]
        Rh[2] = Rh[1]-Hx[1]
        if Rh[0] < 0:
            for i in range(3):
                Rh[i] = -Rh[i]
        return Rh
        
def CentCheck(Cent,H):
    'needs doc string'
    h,k,l = H
    if Cent == 'A' and (k+l)%2:
        return False
    elif Cent == 'B' and (h+l)%2:
        return False
    elif Cent == 'C' and (h+k)%2:
        return False
    elif Cent == 'I' and (h+k+l)%2:
        return False
    elif Cent == 'F' and ((h+k)%2 or (h+l)%2 or (k+l)%2):
        return False
    elif Cent == 'R' and (-h+k+l)%3:
        return False
    else:
        return True
                                    
def GetBraviasNum(center,system):
    """Determine the Bravais lattice number, as used in GenHBravais
    
    :param center: one of: 'P', 'C', 'I', 'F', 'R' (see SGLatt from GSASIIspc.SpcGroup)
    :param system: one of 'cubic', 'hexagonal', 'tetragonal', 'orthorhombic', 'trigonal' (for R)
      'monoclinic', 'triclinic' (see SGSys from GSASIIspc.SpcGroup)
    :return: a number between 0 and 13 
      or throws a ValueError exception if the combination of center, system is not found (i.e. non-standard)

    """
    if center.upper() == 'F' and system.lower() == 'cubic':
        return 0
    elif center.upper() == 'I' and system.lower() == 'cubic':
        return 1
    elif center.upper() == 'P' and system.lower() == 'cubic':
        return 2
    elif center.upper() == 'R' and system.lower() == 'trigonal':
        return 3
    elif center.upper() == 'P' and system.lower() == 'hexagonal':
        return 4
    elif center.upper() == 'I' and system.lower() == 'tetragonal':
        return 5
    elif center.upper() == 'P' and system.lower() == 'tetragonal':
        return 6
    elif center.upper() == 'F' and system.lower() == 'orthorhombic':
        return 7
    elif center.upper() == 'I' and system.lower() == 'orthorhombic':
        return 8
    elif center.upper() == 'A' and system.lower() == 'orthorhombic':
        return 9
    elif center.upper() == 'B' and system.lower() == 'orthorhombic':
        return 10
    elif center.upper() == 'C' and system.lower() == 'orthorhombic':
        return 11
    elif center.upper() == 'P' and system.lower() == 'orthorhombic':
        return 12
    elif center.upper() == 'C' and system.lower() == 'monoclinic':
        return 13
    elif center.upper() == 'P' and system.lower() == 'monoclinic':
        return 14
    elif center.upper() == 'P' and system.lower() == 'triclinic':
        return 15
    raise ValueError('non-standard Bravais lattice center=%s, cell=%s' % (center,system))


    

def SamAng(Tth,Gangls,Sangl,IFCoup):
    """Compute sample orientation angles vs laboratory coord. system

    :param Tth:        Signed theta                                   
    :param Gangls:     Sample goniometer angles phi,chi,omega,azmuth  
    :param Sangl:      Sample angle zeros om-0, chi-0, phi-0          
    :param IFCoup:     True if omega & 2-theta coupled in CW scan
    :returns:  
        psi,gam:    Sample odf angles                              
        dPSdA,dGMdA:    Angle zero derivatives
    """                         
    
    if IFCoup:
        GSomeg = sind(Gangls[2]+Tth)
        GComeg = cosd(Gangls[2]+Tth)
    else:
        GSomeg = sind(Gangls[2])
        GComeg = cosd(Gangls[2])
    GSTth = sind(Tth)
    GCTth = cosd(Tth)      
    GSazm = sind(Gangls[3])
    GCazm = cosd(Gangls[3])
    GSchi = sind(Gangls[1])
    GCchi = cosd(Gangls[1])
    GSphi = sind(Gangls[0]+Sangl[2])
    GCphi = cosd(Gangls[0]+Sangl[2])
    SSomeg = sind(Sangl[0])
    SComeg = cosd(Sangl[0])
    SSchi = sind(Sangl[1])
    SCchi = cosd(Sangl[1])
    AT = -GSTth*GComeg+GCTth*GCazm*GSomeg
    BT = GSTth*GSomeg+GCTth*GCazm*GComeg
    CT = -GCTth*GSazm*GSchi
    DT = -GCTth*GSazm*GCchi
    
    BC1 = -AT*GSphi+(CT+BT*GCchi)*GCphi
    BC2 = DT-BT*GSchi
    BC3 = AT*GCphi+(CT+BT*GCchi)*GSphi
      
    BC = BC1*SComeg*SCchi+BC2*SComeg*SSchi-BC3*SSomeg      
    psi = acosd(BC)
    
    BD = 1.0-BC**2
    C = np.where(BD>1.e-6,rpd/np.sqrt(BD),0.)
    dPSdA = [-C*(-BC1*SSomeg*SCchi-BC2*SSomeg*SSchi-BC3*SComeg),
        -C*(-BC1*SComeg*SSchi+BC2*SComeg*SCchi),
        -C*(-BC1*SSomeg-BC3*SComeg*SCchi)]
      
    BA = -BC1*SSchi+BC2*SCchi
    BB = BC1*SSomeg*SCchi+BC2*SSomeg*SSchi+BC3*SComeg
    gam = atan2d(BB,BA)

    BD = (BA**2+BB**2)/rpd

    dBAdO = 0
    dBAdC = -BC1*SCchi-BC2*SSchi
    dBAdF = BC3*SSchi
    
    dBBdO = BC1*SComeg*SCchi+BC2*SComeg*SSchi-BC3*SSomeg
    dBBdC = -BC1*SSomeg*SSchi+BC2*SSomeg*SCchi
    dBBdF = BC1*SComeg-BC3*SSomeg*SCchi
    
    dGMdA = np.where(BD > 1.e-6,[(BA*dBBdO-BB*dBAdO)/BD,(BA*dBBdC-BB*dBAdC)/BD, \
        (BA*dBBdF-BB*dBAdF)/BD],[np.zeros_like(BD),np.zeros_like(BD),np.zeros_like(BD)])
        
    return psi,gam,dPSdA,dGMdA

BOH = {
'L=2':[[],[],[]],
'L=4':[[0.30469720,0.36418281],[],[]],
'L=6':[[-0.14104740,0.52775103],[],[]],
'L=8':[[0.28646862,0.21545346,0.32826995],[],[]],
'L=10':[[-0.16413497,0.33078546,0.39371345],[],[]],
'L=12':[[0.26141975,0.27266871,0.03277460,0.32589402],
    [0.09298802,-0.23773812,0.49446631,0.0],[]],
'L=14':[[-0.17557309,0.25821932,0.27709173,0.33645360],[],[]],
'L=16':[[0.24370673,0.29873515,0.06447688,0.00377,0.32574495],
    [0.12039646,-0.25330128,0.23950998,0.40962508,0.0],[]],
'L=18':[[-0.16914245,0.17017340,0.34598142,0.07433932,0.32696037],
    [-0.06901768,0.16006562,-0.24743528,0.47110273,0.0],[]],
'L=20':[[0.23067026,0.31151832,0.09287682,0.01089683,0.00037564,0.32573563],
    [0.13615420,-0.25048007,0.12882081,0.28642879,0.34620433,0.0],[]],
'L=22':[[-0.16109560,0.10244188,0.36285175,0.13377513,0.01314399,0.32585583],
    [-0.09620055,0.20244115,-0.22389483,0.17928946,0.42017231,0.0],[]],
'L=24':[[0.22050742,0.31770654,0.11661736,0.02049853,0.00150861,0.00003426,0.32573505],
    [0.13651722,-0.21386648,0.00522051,0.33939435,0.10837396,0.32914497,0.0],
    [0.05378596,-0.11945819,0.16272298,-0.26449730,0.44923956,0.0,0.0]],
'L=26':[[-0.15435003,0.05261630,0.35524646,0.18578869,0.03259103,0.00186197,0.32574594],
    [-0.11306511,0.22072681,-0.18706142,0.05439948,0.28122966,0.35634355,0.0],[]],
'L=28':[[0.21225019,0.32031716,0.13604702,0.03132468,0.00362703,0.00018294,0.00000294,0.32573501],
    [0.13219496,-0.17206256,-0.08742608,0.32671661,0.17973107,0.02567515,0.32619598,0.0],
    [0.07989184,-0.16735346,0.18839770,-0.20705337,0.12926808,0.42715602,0.0,0.0]],
'L=30':[[-0.14878368,0.01524973,0.33628434,0.22632587,0.05790047,0.00609812,0.00022898,0.32573594],
    [-0.11721726,0.20915005,-0.11723436,-0.07815329,0.31318947,0.13655742,0.33241385,0.0],
    [-0.04297703,0.09317876,-0.11831248,0.17355132,-0.28164031,0.42719361,0.0,0.0]],
'L=32':[[0.20533892,0.32087437,0.15187897,0.04249238,0.00670516,0.00054977,0.00002018,0.00000024,0.32573501],
    [0.12775091,-0.13523423,-0.14935701,0.28227378,0.23670434,0.05661270,0.00469819,0.32578978,0.0],
    [0.09703829,-0.19373733,0.18610682,-0.14407046,0.00220535,0.26897090,0.36633402,0.0,0.0]],
'L=34':[[-0.14409234,-0.01343681,0.31248977,0.25557722,0.08571889,0.01351208,0.00095792,0.00002550,0.32573508],
    [-0.11527834,0.18472133,-0.04403280,-0.16908618,0.27227021,0.21086614,0.04041752,0.32688152,0.0],
    [-0.06773139,0.14120811,-0.15835721,0.18357456,-0.19364673,0.08377174,0.43116318,0.0,0.0]]
}

Lnorm = lambda L: 4.*np.pi/(2.0*L+1.)

