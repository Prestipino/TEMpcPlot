import matplotlib.pyplot as plt
import numpy as np

def ransac_lin(data,threshDist,inlierRatio, num=2):
    ''' data: a 2xn dataset with #n data points
        num: the minimum number of points. For line fitting problem, num=2
        n_iter: the number of iterations
        threshDist: the threshold of the distances between points and the fitting line
        inlierRatio: the threshold of the number of inliers express as fraction 80% => 0.8
    '''
    # Plot the data points
    data= np.asanyarray(data)
    number = len(data[1])            # Total number of points
    best_fit = None
    bestInNum = 0                    # Best fitting line with largest number of inliers
    # number of iteration for 99% probability
    n_iter = int(round(np.log(0.01)/np.log(1-inlierRatio**num),0))  
    
    def find_inl(sample):
        vector= sample[:,1]-sample[:,0]
        mod= np.sqrt(vector.dot(vector))
        distance = np.abs((np.cross(vector, data.T-sample[:,0]))/mod) 
        #Compute the inliers with distances smaller than the threshold
        inlierIdx = np.compress(distance<threshDist,
                                data.T,
                                axis=0).T        
        return  inlierIdx      
    
    
    for i in range(n_iter*5):
        # Randomly select 2 points
        sample = data[:,np.random.randint(number, size=num)]
        if  np.sum(sample[:,1]-sample[:,0]) == 0:
            continue
        # Compute the distances between all points with the fitting line       
        inlierIdx = find_inl(sample)
        inlierNum = len(inlierIdx[0])
                
        #Update the number of inliers and fitting model if better model is found 
        if (inlierNum >= round(inlierRatio*number)) and (inlierNum>bestInNum):
            bestInNum = inlierNum
            best_fit = sample[:,:]           
            if inlierNum/number > 0.9:
                break
        #pass
    pass
    if best_fit is None:
        return best_fit  
    inlierIdx = find_inl(best_fit)
    return np.poly1d(np.polyfit(inlierIdx[0],inlierIdx[1],1))
