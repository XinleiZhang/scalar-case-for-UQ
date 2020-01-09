
import numpy as np
import numpy.linalg as la

def EnKF(X,HX,obs,Cdd,Nen):
    XP = X-np.mean(X)
    HXP = HX-np.mean(HX)
    PHT = (1.0 / (Nen - 1.0)) * np.dot(XP, HXP.T)
    HPHT = (1.0 / (Nen - 1.0)) * HXP.dot(HXP.T)
    INV = 1.0 / (HPHT + Cdd)
    KalmanGainMatrix = PHT * INV
    return KalmanGainMatrix

    
def EnKF_MDA(X,HX,obs,Cdd,Nen,alpha):
    XP = X-np.mean(X)
    HXP = HX-np.mean(HX)
    PHT = (1.0 / (Nen - 1.0)) * np.dot(XP, HXP.T)
    HPHT = (1.0 / (Nen - 1.0)) * HXP.dot(HXP.T)
    INV = 1.0 / (HPHT + alpha * Cdd)
    KalmanGainMatrix = PHT * INV
    return KalmanGainMatrix
 
def EnRML(X0,HX0,X,HX,obs,Cdd,Nen):
    XP = X - np.mean(X, keepdims=1) 	
    XP0 = X0 - np.mean(X0)
    HXP = HX - np.mean(HX)
    Cxy= (1.0 / (Nen - 1.0)) * XP.dot(HXP.T)
    Cxxi= (1.0 / (Nen - 1.0)) * XP.dot(XP.T)
    Cxxe= (1.0 / (Nen - 1.0)) * XP0.dot(XP0.T)
    Gen = HXP.dot(la.pinv(XP.reshape(1, len(XP))))
    senmat=Cxxe * Gen.T
    Cyyi = np.dot(Gen * Cxxe, Gen.T) 
    INV = 1. / (Cyyi + Cdd)
    GNGainMatrix = senmat * INV
    penalty= senmat * INV * (Gen * (X - X0))
    return GNGainMatrix, penalty
