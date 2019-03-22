# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 09:03:44 2018

@author: sweener
"""

import numpy as np

def pixels_Vacuum_Vessel():

    '''
    Here we will find the pixel coordinates of the upper and lower outboard
    midplane tile panel boundaries, and the boundary between the lower inclined
    tile panel and the shelf. 
    '''
    
    # Note that the origin is the top-left corner of the image
    xUpOMBound = np.array([70,   193,    212, 253,    159])
    yUpOMBound = np.array([129,  115,    112, 105,    123])
    
    
    xLowOMBound = np.array([54,      109,    189,     220, 250])
    yLowOMBound = np.array([228,    228,     234,     235, 238])
    
    xShelfBound = np.array([91, 127, 229,     180])
    yShelfBound = np.array([310, 317, 352,     331])
    
    xInboard = np.array([22, 22, 22, 22])
    yInboard = np.array([128, 197, 84, 343])
    
#    xUpOMBound = np.array([70,   253])
#    yUpOMBound = np.array([129,  108])
#    
#    
#    xLowOMBound = np.array([54,  250])
#    yLowOMBound = np.array([228, 241])
#    
#    xShelfBound = np.array([91, 229])
#    yShelfBound = np.array([310, 354])    

    return xUpOMBound, yUpOMBound, xLowOMBound, yLowOMBound, xShelfBound, yShelfBound, xInboard, yInboard