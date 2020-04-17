import numpy as np

def feat2HomoConstraints(feat1_, feat2_):
    numPts = feat1_.shape[0]
    a = np.concatenate([feat2_[:,0:1]*feat1_[:,0:1],
                        feat2_[:,0:1]*feat1_[:,1:2],
                        feat2_[:,0:1]*feat1_[:,2:3]], axis =1) 
    M1 = np.concatenate([feat1_[:,0:3], np.zeros([numPts,3]), -a], axis =1)


    b = np.concatenate([feat2_[:,1:2]*feat1_[:,0:1],
                        feat2_[:,1:2]*feat1_[:,1:2],
                        feat2_[:,1:2]*feat1_[:,2:3]], axis =1) 
    M2 = np.concatenate([np.zeros([numPts,3]), feat1_[:,0:3],  -b], axis =1)

    A = np.concatenate([M1,M2], axis =0)
    
    return A 


def hartley_normalization(data): 
    # first scale the data by its standard deviation and center it. This is expressed
    # in matrix form by T
    m = np.mean(data,axis=0)
    s = np.sqrt(2) / np.std(data[:,:2])
    T = np.array([[s,0,-s*m[0]],[0,s,-s*m[1]],[0,0,1]])
    return np.array(T@data.transpose()).transpose(), T

def estHomography(A):

    [u,s,vd] = np.linalg.svd(A)
    H_ = vd[8,:]
    H_est = np.array([[H_[0], H_[1], H_[2]],
                       [H_[3], H_[4], H_[5]],
                       [H_[6], H_[7], H_[8]]])
    return H_est
