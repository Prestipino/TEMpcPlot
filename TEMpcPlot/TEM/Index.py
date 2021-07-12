import numpy as np

from . import math_tools as mt
import itertools
# import  scipy.optimize  as opt


def Peak_find_vectors(Peaks, toll=0.087):
    """ Finds the first or 2 first  smallest non colinear vectors
    for each peak in an image
    Input :
        Peaks array Nx2
        toll (float) tolerance

    Output :
       2*2 or 2*1 array containing the vectors found
    """
    vectors = Peaks[1:] - Peaks[0]  # (2*N matrix) create vectors from 1 peak
    # compute the modulus of each (2*1) vector from the 2*N matrix into a 1*N array
    minarg = np.argsort(mt.mod(vectors))
    for vect_m in minarg[1:]:
        # angle between 2 vectors in radian
        vangle = mt.angle_between_vectors(vectors[minarg[0]], vectors[vect_m])
        if (vangle < np.pi - toll) and (vangle > toll):  # colinearity check
            return vectors[[minarg[0], vect_m]]
    return np.array([vectors[minarg[0]]])


def find_all_2vectors(Peaks, toll=5):
    """
    Finds for each peaks the first 2 smallest non collinear vectors 

    Input :
    - Peaks : Nx2 list containing image peaks coordinates 
    - toll : precision (default number is 5%)

    Output :
    - vectors : n*2 array of vectors sorted by modulus
    """
    atoll = toll * mt.rpd
    vectors = []  # first 2 smallest non colinear vectors
    for i in range(len(Peaks) - 2):
        xx = Peak_find_vectors(Peaks[i:], atoll)
        # finds 2 smallest non colinear vectors for each peak
        vectors.extend(xx)
    return check_colinearity(np.array(vectors), toll_angle=5)


def check_sums(a, b):
    """
    Check whether a linear combinaison of 2 vectors
    can is shorter than the originals

    Input :
    - a , b : 2*1 column vectors

    Output :
    - vector : 2*1 array sorted by modulus

    """
    vector = np.array([a, b, a + b, a - b]) 
    mods = np.argsort(mt.mod(vector))[:2]
    return vector[mods]


def sort_LayerCalib(Peaks, vects, toll=0.1):
    """
    Check if a set of vectors can reindex the peaks projected into its basis

    Input :
    - Peaks is always a row vector
    - Vects a row vector
    - toll : calibration tolerance
    """
    n_index = []
    z = mt.norm(np.cross(*vects[:2]))
    bases = list(itertools.combinations(vects, 2))
    for i_vect in bases:
        npos = mt.change_basis(Peaks, np.vstack([z, *i_vect]).T)
        n_index.append(np.sum(mt.rest_int(npos, toll)))
    argsm = np.argmax(n_index)
    return np.vstack(bases[argsm])


def Find_2D_uc(Peaks, toll_angle=5, toll=0.10):
    """
    Finds the best fitting unit cell for an image

    Input :
    SeqIma : sequence of images

    Output :
    - out : array of unit vectors of length : number_of_images_in_sequence*2
    """
    unit_vectors = find_all_2vectors(Peaks, toll_angle)
    vecti = sort_LayerCalib(Peaks, unit_vectors, toll)
    return check_sums(*vecti)


################################################################


def sort_Calib(Peaks, vects, toll=0.1, toll_angle=5):

    """
    Check if a set of vectors can reindex the peaks projected into its basis

    Input :
    - Peaks is always a row vector
    - Vects a row vector
    - toll : calibration tolerance
    """
    n_index = []
    bases = list(itertools.combinations(vects, 3))
    for i_vect in bases:
        b = mt.angle_between_vectors(np.cross(i_vect[0], i_vect[1]), i_vect[0])
        if  (b - np.pi / 2) > toll_angle * mt.rpd:
            npos = mt.change_basis(Peaks, np.vstack(i_vect).T)
            n_index.append(np.sum(mt.rest_int(npos, toll)))
        else:
            n_index.append(-1)
    argsm = np.argmax(n_index)
    return np.vstack(bases[argsm])


def check_colinearity(vectors, toll_angle=5):
    """ remove all the colinear vectors with longher module 
        the output is ordered by mod
    """
    toll = np.radians(toll_angle)
    vectors3D = []
    vectors = np.array(vectors)[mt.mod(vectors).argsort()][::-1]
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            ang3D = mt.angle_between_vectors(vectors[i], vectors[j])
            if (ang3D > (np.pi - toll)) or (ang3D < toll):  # if colinear
                break
        else:
            vectors3D.append(vectors[i])
    return np.array(vectors3D)[::-1]


def check_3D_coplanarity(redcell, toll_angle=5):
    """
    check the linear combination of 3 vectors in 3D space i.e. if they are coplanar

    Input :
    - redcell : reduced cell if possible by using sort_3D_vect (n*3 row vectors)
    - tol : tolerance in degree when testing coplanarity

    Output :
    - cell : 3*3 row vectors containing 3 non coplanar cell vectors
    """
    b = np.cross(redcell[0], redcell[1])
    for third in redcell[2:]:
        c = mt.angle_between_vectors(b, third)
        if abs(c - np.pi / 2) > toll_angle * mt.rpd:
            return np.array([redcell[0], redcell[1], third])
    else:
        raise ValueError('less than 3 linearly independent vectors')


def check_3Dlincomb(vectors):
    """
    Check whether a linear combinaison is shorter

    Input : 
    - vectors : n*3 column vectors 

    Output :
    - y : x*2 column vectors filtered from linear combinaison

    """
    vect = list((vectors[:]))
    combinations = [(0,1), (0,2), (1,2)]
    while True:
        for i, j in combinations:  # for each vectors except the 1st 2
            vec = check_sums(vect[i], vect[j])        
            if any(vect[i] != vec[0]):
                vect[i] = vec[0]
                if any(vect[j] != vec[1]):
                   vect[j] = vec[1]
                break
            if any(vect[j] != vec[1]):
                vect[j] = vec[1]
                break                
        else:
            break
    return np.array(vect)



def search_twofold(base, toll_angle):
    """
    base are columns vectors for the base
    """
    toll = toll_angle * mt.rpd

    rMT = np.linalg.inv(np.dot(base.T, base))

    zz = np.mgrid[-2:3, -2:3, 0:3]
    uvw = zz.reshape(3, -1).T

    #remove not coprime indices
    xx =np.gcd.reduce(uvw, axis=1)
    uvw = uvw[xx== 1]

    # remove collinear vectors
    UVW = []
    uvwlen = len(uvw)
    for i in range(uvwlen):
        for j in range(i+1, uvwlen):
            if all(np.cross(uvw[i],uvw[j]) == 0):
                break
        else:
            UVW.append(uvw[i])
            
    uvw = np.array(UVW)
    zz = uvw @ uvw.T
    couples =np.where((zz == 2)|(zz==1))