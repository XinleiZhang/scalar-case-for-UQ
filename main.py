#!/usr/bin/env python3

import numpy as np
import analysis
import numpy.linalg as la
import pdb
import os


def forwardmodel(X, q):
    y = 1. + np.sin(np.pi * X) + q
    return y

# user-defined variables

casename = 'E4_EnRML_iter30'
ensemblemethod= 'EnRML' #'EnKF_MDA' 'EnKF'

r = 0.5 # step parameter for EnRML

sigmap = 0.1    # std of prior x
sigmaq = 0.03   # std of model error
sigmad = 0.1    # std of observation error

Nen = 10**4     # ensemble size
maxiter = 30    # maximum iteration number
cri = 1.e-10    # convergence criteria


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

if not os.path.exists('./postprocessing'):
    os.mkdir('./postprocessing')

if not os.path.exists('./postprocessing/'+casename):
    os.mkdir('./postprocessing/'+casename)

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
        pertObs = np.random.normal(0., sigmad, Nen)
        obs =1.+ np.sqrt(alpha)* pertObs
        analysismatrix = analysis.EnKF_MDA(x, y, obs,Cdd,Nen,alpha)
        dx = analysismatrix*(obs-y)
        x = x + dx
        if abs(np.mean(dx)) < cri: converge_flag = 'True'

    elif ensemblemethod=='EnRML':
        obs = 1. + pertObs
        GNGainMatrix, penalty = analysis.EnRML(x0, y0, x, y, obs,Cdd,Nen)
        dx = penalty - GNGainMatrix * (y - obs)
        x = r*x0 + (1-r)*x + r * dx
        if abs(np.mean(dx)) < cri: converge_flag = 'True'

    if converge_flag=='True':
        #pdb.set_trace()
        print ('reach convergence condition at iteration', iter )
        break
    if iter == (maxiter-1): print ('reach max iteration')


np.savetxt('./postprocessing/'+casename+'/posterior_x.txt',x)
np.savetxt('./postprocessing/'+casename+'/posterior_y.txt',y)
np.savetxt('./postprocessing/'+casename+'/prior_x.txt', x0)
np.savetxt('./postprocessing/'+casename+'/prior_y.txt', y0)
