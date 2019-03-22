# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 09:52:15 2018

@author: sweener
"""
import numpy as np
from change_To_Camera_Angles import change_To_Camera_Angles
from pixels_Vacuum_Vessel import pixels_Vacuum_Vessel
from pixels_To_Image_Coords import pixels_To_Image_Coords
from dist_Points_To_Line import dist_Points_To_Line
import matplotlib.pyplot as plt

def camera_Alignment_Error(FitPars):
    sigC = FitPars[0]
    gamma = FitPars[1]
    azim = FitPars[2]
    scaledHeight = FitPars[3]
    horizOffset = FitPars[4]
    vertOffset = FitPars[5]
    Rc = FitPars[6]
    phiMach = FitPars[7]
    Zc = FitPars[8]

    totalXPixels = 384.
    totalYPixels = 384.
    
    ROutWall = 2.34
    phiOutWall = np.linspace(0., 2*np.pi, num=100)
    XOutWall = ROutWall*np.cos(phiOutWall)
    YOutWall = ROutWall*np.sin(phiOutWall)
    ZOutWall = np.linspace(0., 0., num=100) -0.39
    xLowOMBound, yLowOMBound = change_To_Camera_Angles(XOutWall,YOutWall,ZOutWall, sigC, gamma, azim, Rc, phiMach, Zc)
    
    ROutWall = 2.34
    phiOutWall = np.linspace(0., 2*np.pi, num=100)
    XOutWall = ROutWall*np.cos(phiOutWall)
    YOutWall = ROutWall*np.sin(phiOutWall)
    ZOutWall = np.linspace(0., 0., num=100) + 0.42
    xUpOMBound, yUpOMBound = change_To_Camera_Angles(XOutWall,YOutWall,ZOutWall, sigC, gamma, azim,  Rc, phiMach, Zc)
    
    ROutWall = 2.135
    phiOutWall = np.linspace(0., 2*np.pi, num=100)
    XOutWall = ROutWall*np.cos(phiOutWall)
    YOutWall = ROutWall*np.sin(phiOutWall)
    ZOutWall = np.linspace(0., 0., num=100) -0.973
    xShelfBound, yShelfBound = change_To_Camera_Angles(XOutWall,YOutWall,ZOutWall, sigC, gamma, azim,  Rc, phiMach, Zc)
    
    # we will now generate a number of inboard wall loops, take their maximum, and generate a curve
    RInWall = 1.016
    phiInWall = np.linspace(0., 2*np.pi, num=100)
    XInWall = RInWall*np.cos(phiInWall)
    YInWall = RInWall*np.sin(phiInWall)
    numLoops = 100
    xInboard = np.linspace(0., 0., num=numLoops)
    yInboard = np.linspace(0., 0., num=numLoops)
    
    for i in range(0,numLoops):
        ZInWall = np.linspace(0., 0., num=100) -1. + i*2./(numLoops-1.)
        
        xIm, yIm = change_To_Camera_Angles(XInWall,YInWall,ZInWall, sigC, gamma, azim, Rc, phiMach, Zc)  
        maxInd = np.argmax(xIm)
        xInboard[i] = xIm[maxInd]
        yInboard[i] = yIm[maxInd]
    
    
    xUpOMBound2, yUpOMBound2, xLowOMBound2, yLowOMBound2, xShelfBound2, yShelfBound2, xInboard2, yInboard2 = pixels_Vacuum_Vessel()
    
    xUpOMBound3, yUpOMBound3 = pixels_To_Image_Coords(xUpOMBound2, yUpOMBound2, totalXPixels, totalYPixels, scaledHeight*totalXPixels/totalYPixels,scaledHeight, horizOffset, vertOffset)
    xLowOMBound3, yLowOMBound3 = pixels_To_Image_Coords(xLowOMBound2, yLowOMBound2, totalXPixels, totalYPixels, scaledHeight*totalXPixels/totalYPixels,scaledHeight, horizOffset, vertOffset)
    xShelfBound3, yShelfBound3 = pixels_To_Image_Coords(xShelfBound2, yShelfBound2, totalXPixels, totalYPixels, scaledHeight*totalXPixels/totalYPixels,scaledHeight, horizOffset, vertOffset)
    xInboard3, yInboard3 = pixels_To_Image_Coords(xInboard2, yInboard2, totalXPixels, totalYPixels, scaledHeight*totalXPixels/totalYPixels,scaledHeight, horizOffset, vertOffset)
    
#    plt.figure()
#    plt.plot(xUpOMBound3, yUpOMBound3)
#    plt.plot(xUpOMBound, yUpOMBound)
    
    errUpOMBound = dist_Points_To_Line(xUpOMBound3, yUpOMBound3, xUpOMBound, yUpOMBound)
    errLowOMBound = dist_Points_To_Line(xLowOMBound3, yLowOMBound3, xLowOMBound, yLowOMBound)
    errShelfBound = dist_Points_To_Line(xShelfBound3, yShelfBound3, xShelfBound, yShelfBound)
    errInboard = dist_Points_To_Line(xInboard3, yInboard3, xInboard, yInboard)
    
    totalError = np.sqrt(errUpOMBound**2 + errLowOMBound**2 + errShelfBound**2 + errInboard**2)
    

    return totalError
    
    
    
    
    
    
    
    
    