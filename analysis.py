
import numpy as np
import numpy.linalg as la



def EnKF(X,HX,obs,Cdd,Nen):
    XP = X-np.mean(X)
    HXP = HX-np.mean(HX)
    PHT = (1.0 / (Nen - 1.0)) * np.dot(XP, HXP.T)
    #pdb.set_trace()
    HPHT = (1.0 / (Nen - 1.0)) * HXP.dot(HXP.T)

    #if (conInv > 1e16):
    #    print "!!! warning: the matrix (HPHT + R) are singular, inverse would be failed"
    INV = 1./(HPHT + Cdd)
    #INV = INV.A #convert np.matrix to np.ndarray
    #pdb.set_trace()
    KalmanGainMatrix = PHT*INV
    return KalmanGainMatrix

    
def EnKF_MDA(X,HX,obs,Cdd,Nen,alpha):
    XP = X-np.mean(X)
    HXP = HX-np.mean(HX)
    PHT = (1.0 / (Nen - 1.0)) * np.dot(XP, HXP.T)
    #pdb.set_trace()
    HPHT = (1.0 / (Nen - 1.0)) * HXP.dot(HXP.T)

    #if (conInv > 1e16):
    #    print "!!! warning: the matrix (HPHT + R) are singular, inverse would be failed"
    INV = 1./(HPHT + alpha*Cdd)
    #INV = INV.A #convert np.matrix to np.ndarray
    #pdb.set_trace()
    KalmanGainMatrix = PHT*INV
    return KalmanGainMatrix
 
def EnRML(X0,HX0,X,HX,obs,Cdd,Nen):
    XP = X-np.mean(X) 	
    XP0 = X0-np.mean(X0)
    HXP = HX-np.mean(HX)
    
    Cxy= (1.0 / (Nen - 1.0)) *XP.dot(HXP.T)
    #pdb.set_trace()
    Cxxi= (1.0 / (Nen - 1.0)) *XP.dot(XP.T)
    #pdb.set_trace()
    Cxxe= (1.0 / (Nen - 1.0)) *XP0.dot(XP0.T)
    Gen = HXP.dot(la.pinv([XP]))
    senmat=Cxxe*Gen.T
    #pdb.set_trace()
    Cyyi = np.dot(Gen * Cxxe , Gen.T) 
    INV = 1./(Cyyi + Cdd)
    GNGainMatrix = senmat*INV
    penalty= senmat*INV*(Gen*(X-X0))
    return GNGainMatrix, penalty

##############################################################################                        
