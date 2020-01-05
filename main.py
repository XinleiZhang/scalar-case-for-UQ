
# -*- coding: utf-8 -*-
"""
Created on Sep. 07 2018
simple forword model
"""


import numpy as np
import analysis
import numpy.linalg as la
import pdb
import os

def forwardmodel(X, q):
    y = 1. + np.sin(np.pi * X) + q
    return y
#M:forwardmodel E:10^ iter:maxiteration; 
casename = 'M1_E4_EnRML_iter30_NRobs'
ensemblemethod= 'EnRML'
#ensemble parameter
r = 0.5

#determined parameters
sigmap = 0.1
sigmaq = 0.03
sigmad = 0.1

Nen = 10**4                  #Ensemble size
maxiter = 30                #maximum iteration number
cri = 1.e-10

Cxx = sigmap*sigmap
Cqq = sigmaq*sigmaq
Cdd = sigmad*sigmad


x0 = np.random.normal(0, sigmap, Nen)
q = np.random.normal(0, sigmaq, Nen)
y0 = forwardmodel(x0, q)
pertObs = np.random.normal(0., sigmad, Nen)
x = x0
beta = np.zeros(Nen)
converge_flag = 'False'
for iter in range(maxiter):
    y = forwardmodel(x,q)
    
    if ensemblemethod == 'EnKF':
        obs = np.random.normal(1., sigmad, Nen)
        analysismatrix = analysis.EnKF(x, y, obs,Cdd,Nen)
        dx = analysismatrix*(obs-y)
        x = x + dx
        #pdb.set_trace()
        if abs(np.mean(dx)) < cri: converge_flag = 'True'
    
    elif ensemblemethod== 'EnKF_MDA':
        alpha = maxiter
        pertObs = np.random.normal(0., sigmad, Nen)# note: need perturb Obs for each iter 
        obs =1.+ np.sqrt(alpha)* pertObs
        analysismatrix = analysis.EnKF_MDA(x, y, obs,Cdd,Nen,alpha)
        dx = analysismatrix*(obs-y)
        x = x + dx
        if abs(np.mean(dx)) < cri: converge_flag = 'True'

    elif ensemblemethod=='EnRML':
        #pertObs = np.random.normal(0., sigmad, Nen)
        obs = 1. + pertObs # note: fix pertObs can lead to better result
        GNGainMatrix, penalty = analysis.EnRML(x0, y0, x, y, obs,Cdd,Nen)
        dx = penalty - GNGainMatrix * (y - obs)
        x = r*x0 + (1-r)*x + r * dx
        if abs(np.mean(dx)) < cri: converge_flag = 'True'

    elif ensemblemethod=='EnVar':
        epsilon = 0.01
        U = np.eye(Nen)
        pertObs = np.random.normal(0., sigmad, Nen)
        obs = 1. + pertObs
        A0 = x0-np.mean(x0)
        dx = A0.dot(beta)
        xmean = np.mean(x0) + dx
        x = xmean + epsilon*(x0 - np.mean(x0))
        beta, Hes = analysis.EnVar(beta, x, y, obs, Cdd, Nen, epsilon)
        if iter == (maxiter -1):
            eigenv, eigenvec= la.eig(Hes)
            eigenvnew = np.diag(1./(np.sqrt(eigenv)))
            Hes_invroot= np.dot(eigenvec.dot(eigenvnew), la.inv(eigenvec)) 
            x = abs(xmean + np.sqrt(Nen-1)* np.dot(A0.dot( Hes_invroot),U))
            y = forwardmodel(x, q)
    if converge_flag=='True':
        #pdb.set_trace()
        print ('reach convergence condition at iteration', iter )
        break
    if iter == (maxiter-1): print ('reach max iteration')

    if not os.path.exists('./postprocessing/'+casename):
        os.mkdir('./postprocessing/'+casename)

np.savetxt('./postprocessing/'+casename+'/posterior_x.txt',x)
np.savetxt('./postprocessing/'+casename+'/posterior_y.txt',y)
np.savetxt('./postprocessing/'+casename+'/prior_x.txt', x0)
np.savetxt('./postprocessing/'+casename+'/prior_y.txt', y0)
