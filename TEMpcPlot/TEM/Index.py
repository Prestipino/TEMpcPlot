import numpy as np

from . import math_tools as mt
# import  scipy.optimize  as opt










def Peak_find_vectors(Peaks, toll=0.087):
    """ Finds the first or 2 first  smallest non colinear vectors for each peak in an image
    Input : 
        Peaks array Nx2
        toll (float) tolerance

    Output :
       2*2 or 2*1 array containing the vectors found 
    """
    vectors = Peaks[1:] - Peaks[0] #(2*N matrix) create each vector from 1 peak 
    minarg = np.argsort(mt.mod(vectors)) # compute the modulus of each (2*1) vector from the 2*N matrix into a 1*N array
    for vect_m in minarg[1:]:
        vangle = mt.angle_between_vectors(vectors[minarg[0]], vectors[vect_m]) #angle between 2 vectors in radian
        if  (vangle < np.pi-toll) and (vangle > toll): # colinearity check
            return vectors[[minarg[0], vect_m]]
    return np.array([vectors[minarg[0]]])


def find_all_2vectors(Peaks, toll=5):
    """
    Finds for each peaks the first 2 smallest non collinear vectors 
    
    Input :
    - Peaks : 2*N list containing image peaks coordinates 
    - toll : precision (default number is 5%)
    
    Output :
    - vectors : n*2 array of vectors sorted by modulus
    """
    atoll = 5 * mt.rpd
    Peaks = np.array(Peaks).T
    if Peaks.shape[0] ==2:
        Peaks = Peaks.T
    vectors = list(Peak_find_vectors(Peaks, atoll)) #first 2 smallest non colinear vectors
    for i in range(1,len(Peaks)-2):
        vec = Peak_find_vectors(Peaks[i:], atoll) #finds 2 smallest non colinear vectors for each peak
        for vec_i in vec:
            addv = True
            for k, k_vectors in enumerate(vectors): #test for every vector in vectors for each vec_i in vec
                angle1 = mt.angle_between_vectors(vec_i, k_vectors) 
                if (angle1>np.pi-atoll) or (angle1<atoll):   # if is colinear
                    addv = False 
                    if mt.mod(vec_i) < mt.mod(k_vectors): # if the vector is colinear and smaller than the other
                        vectors[k] = vec_i #replace with smallest
                    break
            if addv: # if not colinear add vector
                vectors.append(vec_i)
                    
    vec_mod = mt.mod(np.array(vectors)) # computes "a" vectors (n*2) array modulus into a (n*1) array
    return np.array(vectors)[np.argsort(vec_mod)] 
            
def check_projection(vect, a, b):
    """
    
    Check if the vector projected into (a,b) basis is a multiple of that basis
    
    Input : 
    - vect : column vector (1*2)
    - a , b : (2*1) row vectors
    
    Output :
    - z : 1D array of length 2 containing the remainder of the scalar vector projection with respect to basis
    
    """
    bas = np.array([a,b]).T # (2*2) base vectors 
    z = np.array(mt.project_v(vect, bas) % 1) # 1D array containing the remainder of the vector projection into basis
    z = np.where(z < 0.5, z , 1-z) # convert remainder into number <0.5
    return z

def check_lin_comb(vectors, tol = 0.1):
    """
    Check whether one vector is a linear combinaison of the first 2 smallest vectors
    
    Input : 
    - vectors : n*2 column vectors 
    - tolerance : precision scale (default number = 0.05)
    
    Output :
    - y : x*2 column vectors filtered from linear combinaison
    
    """
    y = [vectors[0], vectors[1]] #first 2 smallest vectors
    for i in vectors[2:]: # for each vectors except the 1st 2
        z = check_projection(i, vectors[0], vectors[1])
        if (z>tol).all() : # keeps only the vectors not being a linear combinaison from other vectors
            np.append(y,i)
    return y

def check_calib(ImageO, vects, toll):
    """
    Check if a set of vectors can reindex the peaks projected into its basis
    
    Input :
    - Peaks is always a row vector
    - Vects a row vector
    - toll : calibration tolerance 
    """
    Peaks = np.array(ImageO.Peaks).T - np.array(ImageO.center)
    n_index = []
    for i_vect in vects :
        proj = (mt.project_v(Peaks, i_vect)).T #Peaks projection into i_vect basis
        n_index.append(np.sum(mt.rest_int(proj, toll))) #filter the peaks whose remainder from the projection > tolerance
    argsm = np.argsort(n_index)
    return np.array(vects)[argsm[-2:]]
        

def check_sums(a,b):
    """
    Check whether a linear combinaison of 2 vectors can is shorter than the originals
    
    Input :
    - a , b : 2*1 column vectors
    
    Output :
    - vector : 2*1 array sorted by modulus
    
    """
    vector = np.array([a, b, a + b, a - b]) # list containing a,b and its sum / difference
    mods = np.argsort([mt.mod(i) for i in vector])[:2]    
    return vector[mods]


def Find_unit_cell(ima, toll_angle=5, toll_index=0.10):
    """
    Finds the best fitting unit cell given a sequence of images
    
    Input : 
    
    SeqIma : sequence of images
    
    Output :
    - out : array of unit vectors of length : number_of_images_in_sequence*2
    
    """      
    unit_vectors = find_all_2vectors(ima.Peaks, toll_angle)  
    unit_vectors = check_lin_comb(unit_vectors, toll_index)
    vecti = check_calib(ima, unit_vectors , toll_index)
    return check_sums(*vecti)

def sort_3D_vect(SeqIma, cell, sort_by_dist= True):
    """
    Sort 3D cell vectors by number of peaks indexed (by default) or by distance only (option)
    
    Input :
    - SeqIma : sequence of images 
    - cell : sequence of images cell vectors found by using col_3D (row vectors)
    - sort_by_dist : boolean 
    
    Output :
    - n_index : list of number of peaks reindexed by cell vector
    - sort : list of cell vectors index sorted by indexing number
    - sort_dist : list of cell vectors index sorted by shortest length
    """
    
    allpos = np.vstack(SeqIma.EwP.pos)
    n_index = []
    n_index_dist = []
    for i in cell :
        proj = (mt.project_v(allpos, i)).T #Peaks projection into i_vect basis
        n_index.append(np.sum(mt.rest_int(proj,0.1))) #number of peaks whose remainder from the projection > tolerance
        if sort_by_dist :
            n_index_dist.append(mt.mod(i))   
    sort = np.argsort(n_index)[::-1] #sort vectors by highest number of indexed peaks
    sort_dist = np.argsort(n_index_dist) #sort vectors by shortest length
    return n_index, sort, sort_dist
    
def check_3D_lin_comb(redcell, tol_degree=5):
    """
    check the linear combination of 3 vectors in 3D space i.e. if they are coplanar
    
    Input :
    - redcell : reduced cell if possible by using sort_3D_vect (n*3 row vectors)
    - tol : tolerance in degree when testing coplanarity
    
    Output :
    - cell : 3*3 row vectors containing 3 non coplanar cell vectors
    """
    b = np.cross(redcell[0], redcell[1])
    for i in range(0, len(redcell)):
        angle = (mt.angle_between_vectors(b, redcell[i]))*180/np.pi
        if angle > 90+tol_degree or angle < 90-tol_degree : #if the 3 vectors are coplanar
            cell = np.array([redcell[0], redcell[1], redcell[i]])
            break
    return cell