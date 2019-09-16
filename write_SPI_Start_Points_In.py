# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:30:44 2019

@author: Ryan
"""

import numpy as np
from Util.write_2D_Data_To_File import write_2D_Data_To_File

"""
For the TRIP3D start.points.in, we need to convert the SPI trajectory from R,
phi, Z to r, theta, phi. This function will take the R, phi, Z data, assume a
magnetic axis position (hard coded), do the conversion, and then write to a
file.
"""


def write_SPI_Start_Points_In(R, Phi, Z):
    
#    lengthR = R.size
#    lengthPhi = Phi.size
#    lengthZ = Z.size
    
       
    
    magR = 1.745 #m
    magZ = -0.02 #m
    
    magX = R - magR
    magZ = Z - magZ
    
    outBrdInds = np.where((R - magR) > 0.)
    magX = magX[outBrdInds]
    magZ = magZ[outBrdInds]
    
    # convert to r, theta, Phi (latter already done)
    r = np.sqrt(magX**2 + magZ**2)
    theta = np.arctan2(magZ, magX)
    
    efitBndInds = np.where(r < 0.65)
    r = r[efitBndInds]
    theta = theta[efitBndInds]*180./np.pi
    phi = Phi[efitBndInds]
    
    
    # prepare data for writing
    data = np.array([r[::3], theta[::3], phi[::3]])
    filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/start.points.in'
    
    write_2D_Data_To_File(np.transpose(data), filename, 0)
    
    