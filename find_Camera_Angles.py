# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:21:53 2018

@author: sweener
"""
import numpy as np
from camera_Alignment_Error import camera_Alignment_Error
from scipy.optimize import minimize

'''
    sigC = FitPars[0]
    gamma = FitPars[1]
    azim = FitPars[2]
    scaledHeight = FitPars[3]
    horizOffset = FitPars[4]
    vertOffset = FitPars[5]
    Rc = FitPars[6]
    phiMach = FitPars[7]
    Zc = FitPars[8]
'''    

initFitPars = np.array([105.,2.9, 0., 0.68, 0.18, 0.01, 2.52, 227., -0.14])

                # sigC        gamma        azi   scaled-height hOffset    voffset       Rc        phiMach        Zc 
theseBounds = [(90.,120.), (-20.,20.), (-20., 20.), (0.2,1.5), (0.,0.5), (-0.2, 0.2), (2.45,2.55), (226., 228.), (-0.3, -0.05)]
res = minimize(camera_Alignment_Error, initFitPars, bounds=theseBounds)

print(res.x)
fitPars = res.x

sigC = fitPars[0]
gamma = fitPars[1]
azim = fitPars[2]
scaledHeight = fitPars[3]
horizOffset = fitPars[4]
vertOffset = fitPars[5]
Rc = fitPars[6]
phiMach = fitPars[7]
Zc = fitPars[8]
