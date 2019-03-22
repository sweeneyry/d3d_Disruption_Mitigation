# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 09:33:04 2018

@author: sweener
"""

import numpy as np

def dist_Points_To_Line(PointsX, PointsY, LineX, LineY):
    '''
    Given some PointsX (float), PointsY (float), LineX (array of float), and LineY
    (array of float), find the minimum distance between each point and the line, and
    return the sum of all minimum distances. 
    '''
    
    numPoints = PointsX.size
    totalDist = 0.
    
    for i in range(0,numPoints):
        thisPointX = PointsX[i]
        thisPointY = PointsY[i]
        
        allDist = np.sqrt((LineX - thisPointX)**2 + (LineY - thisPointY)**2)
        thisMinDist = np.amin(allDist)
        totalDist += thisMinDist
    
        
    return totalDist/numPoints