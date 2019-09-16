# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:32:00 2018

@author: sweener
"""

import numpy as np
import matplotlib.pyplot as plt
from Util.read_3D_Data_To_Array import read_3D_Data_To_Array
from change_To_Camera_Angles import change_To_Camera_Angles
from pixels_Vacuum_Vessel import pixels_Vacuum_Vessel
from pixels_To_Image_Coords import pixels_To_Image_Coords
from write_SPI_Start_Points_In import write_SPI_Start_Points_In
import matplotlib


# Plotting properties
colorVes= 'red'

fontSize = 20
font = {'family' : 'normal',
'weight' : 'normal',
'size'   : fontSize}
matplotlib.rc('font', **font) 

totalXPixels = 384.
totalYPixels = 384.

fieldLinesOn=1

deg2Rad = np.pi/180.

# Image manipulation

fitPars = [  1.04984737e+02  , 2.90979302e+00  , 9.77073348e-02 ,  7.31894934e-01,
   2.25412242e-01 ,  6.02497580e-02  , 2.55000000e+00 ,  2.26000000e+02,
  -1.35863831e-01]

#fitPars = [  1.04969437e+02 ,  3.00635195e+00 ,  3.14171445e-01 ,  6.99830017e-01,
#   2.04165702e-01  , 5.70112884e-02 ,  2.55000000e+00 ,  2.26000000e+02,
#  -1.16736186e-01]

sigC = fitPars[0]
gamma = fitPars[1]
azim = fitPars[2]
scaledHeight = fitPars[3]
horizOffset = fitPars[4]
vertOffset = fitPars[5]
Rc = fitPars[6]
phiMach = fitPars[7]
Zc = fitPars[8]
    




# reading in the image, and collecting image information. the strategy here
# will be to manipulate the image to match the field lines and various structures,
# rather than vice versa.

# 172270
#filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/Visible_Camera/Raman_Ryan/172270_1903p974us.png'
#picTime = 1903.974 #ms
#shotnumber = 172270



# 172271
#filename = 'C:/Users/sweener/Documents/Work/DIIID_Experiment/Raman/Visible_Camera/Raman_Ryan/172271_1906p914us.png'
#filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/Visible_Camera/Raman_Ryan/172271_1907p192us.png'



# 172277
#filename = 'C:/Users/sweener/Documents/Work/DIIID_Experiment/Raman/Visible_Camera/Raman_Ryan/172277_1412p781us.png'
#filename = 'C:/Users/sweener/Documents/Work/DIIID_Experiment/Raman/Visible_Camera/Raman_Ryan/172277_1416p555us.png'
#filename = 'C:/Users/sweener/Documents/Work/DIIID_Experiment/Raman/Visible_Camera/Raman_Ryan/172277_1419p830us.png'



#172281
#filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/Visible_Camera/Raman_Ryan/172281_1906p895us.png'
#filename = 'C:/Users/sweener/Documents/Work/DIIID_Experiment/Raman/Visible_Camera/Raman_Ryan/172281_1906p839us_highPass.png'


# 172282
filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/Visible_Camera/Raman_Ryan/172282_1906p947us.png'
#filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/Visible_Camera/Raman_Ryan/172282_1907p890us_highPass.png'
#filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/Visible_Camera/Raman_Ryan/172282_1908p001us_highPass.png'
# the pic below agrees pretty well with psi=0.98
#filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/Visible_Camera/Raman_Ryan/172282_1907p613us.png'

# in the pic below, upper field lines look like psi=0.8, while lower blobs look like psi=0.4
#filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/Visible_Camera/Raman_Ryan/172282_1908p001us.png'
picTime = 1908.001 #ms
shotnumber = 172282


# PELLET VELOCITY STUDY
#filename = 'C:/Users/sweener/Documents/Work/DIIID_Experiment/Raman/Visible_Camera/Raman_Ryan/172282_1908p168us.png'
#filename = 'C:/Users/sweener/Documents/Work/DIIID_Experiment/Raman/Visible_Camera/Raman_Ryan/172282_1909p555us.png'



fig = plt.figure(figsize=(10,6))
ax0 = fig.add_subplot(121)
im = plt.imread(filename)
imShape = im.shape
imRows = imShape[0]
imCols = imShape[1]
imWidthOverHeight = imCols/imRows
scaledWidth = scaledHeight*imWidthOverHeight
implot = plt.imshow(im, aspect='equal', extent=(-scaledWidth/2. + horizOffset, 
                                                scaledWidth/2.+ horizOffset, 
                                                -scaledHeight/2. + vertOffset, 
                                                scaledHeight/2.+ vertOffset))

RInWall = 1.016
phiInWall = np.linspace(0., 2*np.pi, num=100)
XInWall = RInWall*np.cos(phiInWall)
YInWall = RInWall*np.sin(phiInWall)

#for i in range(0,10):
#    ZInWall = np.linspace(0., 0., num=100) -1. + i*2./9.
#    
#    xIm, yIm = change_To_Camera_Angles(XInWall,YInWall,ZInWall, sigC, gamma, azim, Rc, phiMach, Zc)
#    
#    plt.plot(xIm, yIm, color=colorVes, linewidth=0.5)
    
ROutWall = 2.34
phiOutWall = np.linspace(0., 2*np.pi, num=100)
XOutWall = ROutWall*np.cos(phiOutWall)
YOutWall = ROutWall*np.sin(phiOutWall)
ZOutWall = np.linspace(0., 0., num=100) -0.39
xIm, yIm = change_To_Camera_Angles(XOutWall,YOutWall,ZOutWall, sigC, gamma, azim,  Rc, phiMach, Zc)
#plt.plot(xIm, yIm, '-', color=colorVes, linewidth=0.5)  

ROutWall = 2.135
phiOutWall = np.linspace(0., 2*np.pi, num=100)
XOutWall = ROutWall*np.cos(phiOutWall)
YOutWall = ROutWall*np.sin(phiOutWall)
ZOutWall = np.linspace(0., 0., num=100) -0.973
xIm, yIm = change_To_Camera_Angles(XOutWall,YOutWall,ZOutWall, sigC, gamma, azim,  Rc, phiMach, Zc)
#plt.plot(xIm, yIm, '-', color=colorVes, linewidth=0.5)  

ROutWall = 2.34
phiOutWall = np.linspace(0., 2*np.pi, num=100)
XOutWall = ROutWall*np.cos(phiOutWall)
YOutWall = ROutWall*np.sin(phiOutWall)
ZOutWall = np.linspace(0., 0., num=100) + 0.42
xIm, yIm = change_To_Camera_Angles(XOutWall,YOutWall,ZOutWall, sigC, gamma, azim,  Rc, phiMach, Zc)
#plt.plot(xIm, yIm, '-', color=colorVes, linewidth=0.5) 


# FIELD LINE PLOTTING -------------------------------------

filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/psi0p4/out.92'
allFieldLines = read_3D_Data_To_Array(filename)
# get the data out of the tuple
allFieldLines = allFieldLines[0]


#for i in np.array([38, 40, 42, 44, 45, 46, 47, 48] ):

if fieldLinesOn:
    numFieldLines = 50
    for i in np.array([40, 43,  45, 47, 49]):
#    for i in range(0,numFieldLines):
        thisFieldLine = allFieldLines[:,:,i]
        R = thisFieldLine[:,0]
        Z = thisFieldLine[:,1]
        phi = thisFieldLine[:,2] - np.pi/1.8
        
        X = R*np.cos(phi)
        Y = R*np.sin(phi)
        
        xIm, yIm = change_To_Camera_Angles(X,Y,Z, sigC, gamma, azim,  Rc, phiMach, Zc)
        
        plt.plot(xIm, yIm, color='orange', linewidth=1)


filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/psi0p8/out.92'
allFieldLines = read_3D_Data_To_Array(filename)
# get the data out of the tuple
allFieldLines = allFieldLines[0]


#[48, 49]

## q=3 field lines
if fieldLinesOn:
    numFieldLines = 50
    for i in np.array([43, 46,  48, 49]):
#    for i in range(0,numFieldLines):
        thisFieldLine = allFieldLines[:,:,i]
        R = thisFieldLine[:,0]
        Z = thisFieldLine[:,1]
        phi = thisFieldLine[:,2] - np.pi/1.8
        
        X = R*np.cos(phi)
        Y = R*np.sin(phi)
        
        xIm, yIm = change_To_Camera_Angles(X,Y,Z, sigC, gamma, azim,  Rc, phiMach, Zc)
        
        plt.plot(xIm, yIm, color='YELLOW', linewidth=1)
    


# SPI TRAJECTORY -------------------------------------
# we will now plot the trajectory of the SPI
phiSPI = 360. - 135.
alphaInj = 0. # -20. # angle relative to normal of the injection
RInit = 2.24
ZInit = 0.72
RFinal = 1.016
ZFinal =  -0.9#-0.4
numPts=201


s = np.linspace(0., RInit - RFinal, num=numPts) # this is in meters
if alphaInj ==0:
    LSPI = s*0.
    PhiP = s*0.
    RSPI = RInit - s
else:
    LSPI = s*np.sin(alphaInj*deg2Rad)  
    PhiP = np.arctan(LSPI/(RInit - s*np.cos(alphaInj*deg2Rad))) # phi of pellet  
    RSPI = LSPI/np.sin(PhiP)    
ZSPI = (ZFinal - ZInit)*np.linspace(0., 1., num=numPts) + ZInit

# write out start points along the SPI trajectory for the TRIP3D field line
# tracing
write_SPI_Start_Points_In(RSPI, PhiP, ZSPI)

distSPI = np.sqrt(s**2 + (ZSPI - ZInit)**2 + LSPI**2)
distInds = np.where(np.round_(distSPI*100., decimals=2) % 10 < 1.0)
XSPI = RSPI*np.cos(PhiP + phiSPI*deg2Rad)
YSPI = RSPI*np.sin(PhiP + phiSPI*deg2Rad)
xIm, yIm = change_To_Camera_Angles(XSPI,YSPI,ZSPI, sigC, gamma, azim,  Rc, phiMach, Zc)
plt.plot(xIm, yIm, color='red', linewidth=2)
plt.plot(xIm[distInds], yIm[distInds], 'o', color='black', markersize=3)
plt.plot(xIm[np.squeeze(distInds)[5]], yIm[np.squeeze(distInds)[5]], 'o', color='magenta', markersize=9)

xUpOMBound2, yUpOMBound2, xLowOMBound2, yLowOMBound2, xShelfBound2, yShelfBound2, xInboard2, yInboard2 = pixels_Vacuum_Vessel()

xUpOMBound3, yUpOMBound3 = pixels_To_Image_Coords(xUpOMBound2, yUpOMBound2, totalXPixels, totalYPixels, scaledHeight*totalXPixels/totalYPixels,scaledHeight, horizOffset, vertOffset)
xLowOMBound3, yLowOMBound3 = pixels_To_Image_Coords(xLowOMBound2, yLowOMBound2, totalXPixels, totalYPixels, scaledHeight*totalXPixels/totalYPixels,scaledHeight, horizOffset, vertOffset)
xShelfBound3, yShelfBound3 = pixels_To_Image_Coords(xShelfBound2, yShelfBound2, totalXPixels, totalYPixels, scaledHeight*totalXPixels/totalYPixels,scaledHeight, horizOffset, vertOffset)
xInboard3, yInboard3 = pixels_To_Image_Coords(xInboard2, yInboard2, totalXPixels, totalYPixels, scaledHeight*totalXPixels/totalYPixels,scaledHeight, horizOffset, vertOffset)
 
#plt.plot(xUpOMBound3, yUpOMBound3, 'o', color='red')
#plt.plot(xLowOMBound3, yLowOMBound3, 'o', color='red')
#plt.plot(xShelfBound3, yShelfBound3, 'o', color='red')
#plt.plot(xInboard3, yInboard3, 'o', color='red')
    
    
plt.xlim(-0.13, 0.36)
plt.ylim(-0.3, 0.4)  

ax0.set_yticklabels([])
ax0.set_xticklabels([])




# Following is specific to efit_172281_1900ms.png
#filename = 'C:/Users/sweener/Documents/Work/DIIID_Experiment/Raman/efit_172281_1900ms.png'
#xPixelUpperCorner = 312 
#yPixelUpperCorner = 64
#xPixelLowerCorner = 312
#yPixelLowerCorner = 665
#totalXPixels = 559
#totalYPixels = 729

# Following is specific to efit_172270_1900ms.png
filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/efit_172270_1900ms.png'
xPixelUpperCorner = 311 
yPixelUpperCorner = 66
xPixelLowerCorner = 311
yPixelLowerCorner = 667
totalXPixels = 540
totalYPixels = 729


# This is a property of the vessel, and is constant
RUpperCorner = 1.419
ZUpperCorner = 1.348

RLowerCorner = 1.42
ZLowerCorner = -1.363

scaleRealToPixel = (yPixelLowerCorner-yPixelUpperCorner)/(ZUpperCorner - ZLowerCorner)

RImageOrigin = RLowerCorner - xPixelLowerCorner/scaleRealToPixel
ZImageOrigin = ZUpperCorner - (totalYPixels - yPixelUpperCorner)/scaleRealToPixel


# we will now plot the trajectory of the SPI

RProj = RInit - s*np.cos(alphaInj*deg2Rad)
distSPI = np.sqrt((RProj - RInit)**2 + (ZSPI - ZInit)**2)
distInds = np.where(np.round_(distSPI*100., decimals=2) % 10 < 1.0)

RSPIPixels = (RProj - RImageOrigin)*scaleRealToPixel
ZSPIPixels = totalYPixels - (ZSPI - ZImageOrigin)*scaleRealToPixel

horizOffset = RImageOrigin
vertOffset = ZImageOrigin
scaledHeight = totalYPixels/scaleRealToPixel
scaledWidth = scaledHeight*totalXPixels/totalYPixels



ax = fig.add_subplot(122)


im = plt.imread(filename)
imShape = im.shape
imRows = np.float(imShape[0])
imCols = np.float(imShape[1])
imWidthOverHeight = imCols/imRows
scaledWidth = scaledHeight*imWidthOverHeight
implot = plt.imshow(im, aspect='equal', extent=(horizOffset, 
                                                scaledWidth+ horizOffset, 
                                                vertOffset, 
                                                scaledHeight + vertOffset))


plt.plot(RProj, ZSPI, color='red')
plt.plot(RProj[distInds], ZSPI[distInds], 'o', color='black', markersize=3)

plt.plot(RProj[np.squeeze(distInds)[5]], ZSPI[np.squeeze(distInds)[5]], 'o', color='magenta', markersize=9)
plt.ylim([-1.5, 1.5])
plt.xlim([0.9, 2.5])
plt.xlabel('$R$ (m)')
plt.ylabel('$Z$ (m)')
plt.xticks(np.arange(0.9, 2.5, step=0.5))
plt.yticks(np.arange(-1.0, 1.1, step=1.))




# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.34, 0.12, 0.15, 0.15]
ax2 = fig.add_axes([left, bottom, width, height])

ax2.plot(XSPI, YSPI, color='red', linewidth=3)
ax2.plot(XOutWall, YOutWall, color='black')
ax2.plot(XInWall, YInWall, color='black')
ax2.plot(Rc*np.cos((360. - phiMach)*deg2Rad), Rc*np.sin((360. - phiMach)*deg2Rad), 'o', color='green')
ax2.set_aspect('equal')
# Turn off tick labels
ax2.set_yticklabels([])
ax2.set_xticklabels([])

plt.text(-2.2, -0.4, 'Top-down\nview', color='black', fontsize=16)
plt.text(-12, 20., np.array2string(np.array(shotnumber)) + '\n' + 
    np.array2string(np.array(picTime)) + ' ms', color='white', fontsize=16)

if fieldLinesOn:
    plt.text(-6, 5, '$q=2$', color='orange', fontsize=20)    
    plt.text(-1, 15, '$q=3$', color='yellow', fontsize=20)    
    
    
    
    
    
    
    
    
    
    
    
