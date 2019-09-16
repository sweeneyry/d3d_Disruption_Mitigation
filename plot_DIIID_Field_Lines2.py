#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:27:05 2019

@author: Ryan
"""


from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from Util.read_3D_Data_To_Array import read_3D_Data_To_Array
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fontSize = 12
font = {'family' : 'normal',
'weight' : 'normal',
'size'   : fontSize}
matplotlib.rc('font', **font) 


PlotFieldLines=0

deg2Rad = np.pi/180.

# taken from write_SPI_Start_Points. THIS IS BAD CODING!!!
magR = 1.76 #1.745 #m
magZ = 0.0 #m

relPhiSX90 = np.array([-315., 45.]) #-20.
relPhiDISRAD = np.array([-75., 285.]) #-20. 

# TEST 1==========================
# COUNTER-IP

filenameP = ['/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Offset_m0p1m/CoIp/out2Transits.12', 
             '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Offset_m0p1m/CtIp/out2Transits.12']

#filenameP = ['/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Nominal_Traj/CtIp/out2Transits.12', 
#             '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Nominal_Traj/CoIp/out2Transits.12']
#filenameP = ['/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Nominal_Traj/CtIp/out2Transits.12']
#filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Nominal_Traj/CoIp/out.92'


# CO-IP
filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Nominal_Traj/CtIp/out.92'
filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/psi0p4/out.92'




# TEST 2==========================
# COUNTER-IP
#relPhiSX90 = -315.
#relPhiDISRAD = -75.
#filenameP = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Upward_Traj/CtIp/out.12'
#filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Upward_Traj/CtIp/out.92'


# CO-IP
#relPhiSX90 = 45.
#relPhiDISRAD = 345.
#filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Upward_Traj/CoIp/out.92'
#filenameP = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Upward_Traj/CoIp/out.12'



# TEST 3==========================
# COUNTER-IP
#relPhiSX90 = -315.
#relPhiDISRAD = -75.
#filenameP = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Downward_Traj/CtIp/out.12'
#filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Downward_Traj/CtIp/out.92'


# CO-IP
#relPhiSX90 = 45.
#relPhiDISRAD = 345.
#filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Downward_Traj/CoIp/out.92'
#filenameP = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Downward_Traj/CoIp/out.12'






#filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/psi0p3/out.92'


allFieldLines = read_3D_Data_To_Array(filename)
# get the data out of the tuple
allFieldLines = allFieldLines[0]

print(np.shape(allFieldLines))

fig = plt.figure(figsize=plt.figaspect(1.)*1.5)
ax = Axes3D(fig)

#ax = fig.gca(projection='3d')

# The SPI trajectory
SPI2R = 2.24
SPI2Z = 0.72
SPI2Phi = np.array(135.*deg2Rad)
SPI2RFinal = 1.016
SPI2ZFinal = -0.9#-0.3
SPI2X = SPI2R*np.cos(-SPI2Phi)
SPI2Y = SPI2R*np.sin(-SPI2Phi)
SPI2XFinal = SPI2RFinal*np.cos(-SPI2Phi)
SPI2YFinal = SPI2RFinal*np.sin(-SPI2Phi)
X = np.array([SPI2X, SPI2XFinal])
Y = np.array([SPI2Y, SPI2YFinal])
Z = np.array([SPI2Z, SPI2ZFinal])
ax.plot(X,Y,Z, color='cyan', linewidth=3)
X = np.array([SPI2X])
Y = np.array([SPI2Y])
Z = np.array([SPI2Z])
ax.plot(X,Y,Z, 'o', color='cyan')

if PlotFieldLines:
    for i in np.array([40, 43,  45, 47, 49]):
        thisFieldLine = allFieldLines[:,:,i]
        R = thisFieldLine[:,0]
        Z = thisFieldLine[:,1]
        phi = thisFieldLine[:,2]
        
        X = R*np.cos(phi- 3*np.pi/4.)
        Y = R*np.sin(phi- 3*np.pi/4.)

#        X = R*np.cos(phi- np.pi/8.)
#        Y = R*np.sin(phi- np.pi/8.)
        
        ax.plot(X,Y,Z, color='g')
    

phi = np.linspace(0, 2.*np.pi)
X = magR*np.cos(phi)
Y = magR*np.sin(phi)
ax.plot(X,Y,magZ, '--', color='black')

#scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz']); ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
#ax.auto_scale_xyz(scaling)



# let's add the SXR arrays to this view ----------------------

zDrift = 0.06
RDrift = -0.03 #-0.1

axuv210Phi = np.array(210.*deg2Rad)
axuv210Rp1UpR = 2.290 - RDrift
axuv210Rp1UpZ = 0.780 - zDrift
axuv210UpX = axuv210Rp1UpR*np.cos(-axuv210Phi)
axuv210UpY = axuv210Rp1UpR*np.sin(-axuv210Phi)
X = np.array([axuv210UpX])
Y = np.array([axuv210UpY])
Z = np.array([axuv210Rp1UpZ])
ax.plot(X,Y,Z, 'o', color='red')


axuv210Rp1LowR = 2.300- RDrift
axuv210Rp1LowZ = 0.740 - zDrift
axuv210LowX = axuv210Rp1LowR*np.cos(-axuv210Phi)
axuv210LowY = axuv210Rp1LowR*np.sin(-axuv210Phi)
X = np.array([axuv210LowX])
Y = np.array([axuv210LowY])
Z = np.array([axuv210Rp1LowZ])
ax.plot(X,Y,Z, 'o', color='red')
X = np.array([axuv210LowX, axuv210LowX])
Y = np.array([axuv210LowY, axuv210LowY])
Z = np.array([axuv210Rp1LowZ, -axuv210Rp1LowZ])
ax.plot(X,Y,Z, color='red')

axuv90Phi = np.array(90.*deg2Rad)
axuv90Rp1UpR = 2.272- RDrift
axuv90Rp1UpZ = 0.781 - zDrift
axuv90UpX = axuv90Rp1UpR*np.cos(-axuv90Phi)
axuv90UpY = axuv90Rp1UpR*np.sin(-axuv90Phi)
X = np.array([axuv90UpX])
Y = np.array([axuv90UpY])
ax.plot(X,Y,axuv90Rp1UpZ, 'o', color='blue')



axuv90Rp1LowR = 2.304- RDrift
axuv90Rp1LowZ = 0.731 - zDrift
axuv90LowX = axuv90Rp1LowR*np.cos(-axuv90Phi)
axuv90LowY = axuv90Rp1LowR*np.sin(-axuv90Phi)
X = np.array([axuv90LowX])
Y = np.array([axuv90LowY])
Z = np.array([axuv90Rp1LowZ])
ax.plot(X,Y,axuv90Rp1LowZ, 'o', color='blue')
X = np.array([axuv90LowX, axuv90LowX])
Y = np.array([axuv90LowY, axuv90LowY])
Z = np.array([axuv90Rp1LowZ, -axuv90Rp1LowZ])
ax.plot(X,Y,Z, color='blue')

thetaInc = np.linspace(np.pi/2, np.pi*(55.2+90.)/180., num=50)

axuvRadius=2.2
xVert = np.concatenate((np.array([axuv90LowX, axuv90LowX]), axuv90LowX + np.cos(thetaInc)*axuvRadius*np.cos(-axuv90Phi), np.array([axuv90LowX])))
yVert = np.concatenate((np.array([axuv90LowY, axuv90LowY]), axuv90LowY + np.cos(thetaInc)*axuvRadius*np.sin(-axuv90Phi), np.array([axuv90LowY])))
zVert = np.concatenate((np.array([axuv90Rp1LowZ, axuv90Rp1LowZ-axuvRadius]), axuv90Rp1LowZ-axuvRadius*np.sin(thetaInc), np.array([axuv90Rp1LowZ])))

vtx = np.array([[xVert], [yVert], [zVert]])

#vtx = np.array([[axuv90LowX, axuv90LowX, axuv90LowX*0.5, axuv90LowX], 
#                [axuv90LowY, axuv90LowY, axuv90LowY*0.5, axuv90LowY],
#                [axuv90Rp1LowZ, -axuv90Rp1LowZ,  axuv90Rp1LowZ,  axuv90Rp1LowZ]])

tri = Poly3DCollection([np.transpose(np.squeeze(vtx))])
tri.set_alpha(0.5)
tri.set_color('blue')
ax.add_collection3d(tri)


thetaInc = np.linspace(np.pi*(60.8+90.)/180., np.pi*(112.8+90.)/180., num=50)

axuvRadius=1.5
xVert = np.concatenate((np.array([axuv90UpX]), axuv90UpX + np.cos(thetaInc)*axuvRadius*np.cos(-axuv90Phi), np.array([axuv90UpX])))
yVert = np.concatenate((np.array([axuv90UpY]), axuv90UpY + np.cos(thetaInc)*axuvRadius*np.sin(-axuv90Phi), np.array([axuv90UpY])))
zVert = np.concatenate((np.array([axuv90Rp1UpZ]), axuv90Rp1UpZ-axuvRadius*np.sin(thetaInc), np.array([axuv90Rp1UpZ])))

vtx = np.array([[xVert], [yVert], [zVert]])

#vtx = np.array([[axuv90LowX, axuv90LowX, axuv90LowX*0.5, axuv90LowX], 
#                [axuv90LowY, axuv90LowY, axuv90LowY*0.5, axuv90LowY],
#                [axuv90Rp1LowZ, -axuv90Rp1LowZ,  axuv90Rp1LowZ,  axuv90Rp1LowZ]])

tri = Poly3DCollection([np.transpose(np.squeeze(vtx))])
tri.set_alpha(0.5)
tri.set_color('blue')
ax.add_collection3d(tri)


thetaInc = np.linspace(np.pi/2, np.pi*(55.2+90.)/180., num=50)
axuvRadius=2.2
xVert = np.concatenate((np.array([axuv210LowX, axuv210LowX]), axuv210LowX + np.cos(thetaInc)*axuvRadius*np.cos(-axuv210Phi), np.array([axuv210LowX])))
yVert = np.concatenate((np.array([axuv210LowY, axuv210LowY]), axuv210LowY + np.cos(thetaInc)*axuvRadius*np.sin(-axuv210Phi), np.array([axuv210LowY])))
zVert = np.concatenate((np.array([axuv210Rp1LowZ, axuv210Rp1LowZ-axuvRadius]), axuv210Rp1LowZ-axuvRadius*np.sin(thetaInc), np.array([axuv210Rp1LowZ])))

vtx = np.array([[xVert], [yVert], [zVert]])



tri = Poly3DCollection([np.transpose(np.squeeze(vtx))])
tri.set_alpha(0.5)
tri.set_color('red')
ax.add_collection3d(tri)


thetaInc = np.linspace(np.pi*(60.8+90.)/180., np.pi*(112.8+90.)/180., num=50)
axuvRadius=1.5
xVert = np.concatenate((np.array([axuv210UpX]), axuv210UpX + np.cos(thetaInc)*axuvRadius*np.cos(-axuv210Phi), np.array([axuv210UpX])))
yVert = np.concatenate((np.array([axuv210UpY]), axuv210UpY + np.cos(thetaInc)*axuvRadius*np.sin(-axuv210Phi), np.array([axuv210UpY])))
zVert = np.concatenate((np.array([axuv210Rp1UpZ]), axuv210Rp1UpZ-axuvRadius*np.sin(thetaInc), np.array([axuv210Rp1UpZ])))

vtx = np.array([[xVert], [yVert], [zVert]])



tri = Poly3DCollection([np.transpose(np.squeeze(vtx))])
tri.set_alpha(0.5)
tri.set_color('red')
ax.add_collection3d(tri)


# and now let's add the interferometers ----------------------
phiR0 = 225. # deg
phiV = 240. # deg
RV1 = 1.48 #m
RV2 = 1.94 #m
RV3 = 2.1 #m
zIntLow = -1.3
zIntHigh = 1.3
RIntOut = 2.4
RIntIn = 1.

XV1 = RV1*np.cos(-phiV*deg2Rad)
YV1 = RV1*np.sin(-phiV*deg2Rad)
X = np.array([XV1, XV1])
Y = np.array([YV1, YV1])
Z = np.array([zIntLow, zIntHigh])
ax.plot(X,Y,Z, color='red')

XV2 = RV2*np.cos(-phiV*deg2Rad)
YV2 = RV2*np.sin(-phiV*deg2Rad)
X = np.array([XV2, XV2])
Y = np.array([YV2, YV2])
Z = np.array([zIntLow, zIntHigh])
ax.plot(X,Y,Z, color='blue')

XV3 = RV3*np.cos(-phiV*deg2Rad)
YV3 = RV3*np.sin(-phiV*deg2Rad)
X = np.array([XV3, XV3])
Y = np.array([YV3, YV3])
Z = np.array([zIntLow, zIntHigh])
ax.plot(X,Y,Z, color='magenta')

X = np.array([RIntOut*np.cos(-phiR0*deg2Rad), RIntIn*np.cos(-phiR0*deg2Rad)])
Y = np.array([RIntOut*np.sin(-phiR0*deg2Rad), RIntIn*np.sin(-phiR0*deg2Rad)])
Z = np.array([0., 0.])
ax.plot(X,Y,Z, color='black')


# plasma boundary
filename = '/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/plasma_Bdry_172270.txt'
RZBdry = np.loadtxt(filename, delimiter=',')
bndInds = np.where(RZBdry[:,0] != 0.)
RZBdry = RZBdry[np.squeeze(bndInds),:]
RBdry = RZBdry[:,0]
ZBdry = RZBdry[:,1]

XBdry = RBdry*np.cos(-axuv210Phi)
YBdry = RBdry*np.sin(-axuv210Phi)
ax.plot(XBdry, YBdry, ZBdry, color='black')

XBdry = RBdry*np.cos(-phiV*deg2Rad)
YBdry = RBdry*np.sin(-phiV*deg2Rad)
ax.plot(XBdry, YBdry, ZBdry, color='black')


XBdry = RBdry*np.cos(-phiR0*deg2Rad)
YBdry = RBdry*np.sin(-phiR0*deg2Rad)
ax.plot(XBdry, YBdry, ZBdry, color='black')

XBdry = RBdry*np.cos(-axuv90Phi)
YBdry = RBdry*np.sin(-axuv90Phi)
ax.plot(XBdry, YBdry, ZBdry, color='black')




XBdry = RBdry*np.cos(-SPI2Phi)
YBdry = RBdry*np.sin(-SPI2Phi)
ax.plot(XBdry, YBdry, ZBdry, color='black')

ax.set_xlim3d(left=-2., right=2.)
ax.set_ylim3d(bottom=-2., top=2.)
ax.set_zlim3d(bottom=-2., top=2.)
ax.view_init(elev=35., azim=160.)
ax.set_axis_off()



# POINCARE PLOTS ======================================================
fig = plt.figure(figsize=(5,8))
ax0 = plt.subplot2grid((2,2), (0,0))
ax0.plot(magR, magZ, 'x', color='black')
plt.xlim([1.,2.5])
plt.ylim([-1.5,1.5])
plt.xlabel('$R$ [m]')
plt.ylabel('$Z$ [m]')
plt.xticks(np.arange(1., 2.1, 1.))
plt.yticks(np.arange(-1.5, 1.6, 1.5))
plt.title('AXUV $+75^\circ$')
#plt.plot([RV1, RV1], [-1.5, 1.5], color='red')
#plt.plot([RV2, RV2], [-1.5, 1.5], color='blue')
#plt.plot([RV3, RV3], [-1.5, 1.5], color='magenta')
#plt.plot([1., 2.5], [0., 0.], color='black')



# initialize second subplot
ax1 = plt.subplot2grid((2,2), (1,0), colspan=2)
plt.ylabel('$\Theta_{inc}$ [deg]')
plt.xlabel('Time after arrival [ms]')
plt.yticks(np.arange(0., 121., 40.))
plt.xticks(np.arange(0., 3.1, 1.))
plt.ylim([0, 125])


# initialize second subplot
ax2 = plt.subplot2grid((2,2), (0,1))
ax2.plot(magR, magZ, 'x', color='black')
plt.xlim([1.,2.5])
plt.ylim([-1.5,1.5])
plt.xlabel('$R$ [m]')
plt.xticks(np.arange(1., 2.1, 1.))
plt.yticks(np.arange(-1.5, 1.6, 1.5)) 
plt.title('AXUV $-45^\circ$') 
ax2.axes.yaxis.set_ticklabels([])


# initialize second subplot
#ax3 = fig.add_subplot(224)
#plt.xlabel('Time (arb)')
#plt.yticks(np.arange(0., 121., 40.))
#plt.xticks(np.arange(0., 19., 6.))
#plt.ylim([0, 120])
#ax3.axes.yaxis.set_ticklabels([])


ax0.set_aspect('equal')
ax2.set_aspect('equal')



for i in range(0,2):
    
    
    allPoints = read_3D_Data_To_Array(filenameP[i])
    # get the data out of the tuple
    allPoints = np.array(allPoints[0])
    
    
    numPts = (allPoints.shape)[2]
    sx90Inds = np.zeros(numPts)
    disradInds = np.zeros(numPts)
    
    
    sx90Ind = np.where(np.abs(allPoints[:,0,0])%360 == np.abs(relPhiSX90[i]))
    disradInd = np.where(np.abs(allPoints[:,0,0])%360 == np.abs(relPhiDISRAD[i]))
    
    
    allSX90Points = np.transpose(np.squeeze(allPoints[sx90Ind[0], :, :]))
    allDISRADPoints = np.transpose(np.squeeze(allPoints[disradInd[0],:, :]))
    
    print('shape', np.shape(allSX90Points))
    
    
    
    rSX90Points = allSX90Points[:,1]
    thetaSX90Points = allSX90Points[:,3]*deg2Rad
    RSX90Points = magR + rSX90Points*np.cos(thetaSX90Points)
    ZSX90Points = magZ + rSX90Points*np.sin(thetaSX90Points)
    
    rDISRADPoints = allDISRADPoints[:,1]
    thetaDISRADPoints = allDISRADPoints[:,3]*deg2Rad
    RDISRADPoints = magR + rDISRADPoints*np.cos(thetaDISRADPoints)
    ZDISRADPoints = magZ + rDISRADPoints*np.sin(thetaDISRADPoints)
    


    
    
    # PLOTTING ==============================  
    if i == 0:
        marker = 'o'
        markerSize = 3
        markerEdgeWidth = 0
        
        # these only need to be plotted once
        ax2.plot(axuv90Rp1UpR, axuv90Rp1UpZ, 's', color='blue')
        ax2.plot(axuv90Rp1LowR,axuv90Rp1LowZ, 's', color='blue')
   
        ax0.plot(axuv210Rp1UpR, axuv210Rp1UpZ, 's', color='red')
        ax0.plot(axuv210Rp1LowR,axuv210Rp1LowZ, 's', color='red') 
        
        # these are specific to counter
        ax2.plot(RSX90Points[1:], ZSX90Points[1:], marker, color='blue', markersize=markerSize, 
                 markeredgewidth=markerEdgeWidth)
        ax0.plot(RDISRADPoints[1:], ZDISRADPoints[1:], marker, color='red', markersize=markerSize, 
                 markeredgewidth=markerEdgeWidth)        
    else:
        marker = 'x'
        markerSize = 3
        markerEdgeWidth = 1
        
        ax2.plot(RSX90Points[1:,0], ZSX90Points[1:,0], marker, color='blue', markersize=5, 
                 markeredgewidth=markerEdgeWidth)
        ax0.plot(RDISRADPoints[1:,0], ZDISRADPoints[1:,0], marker, color='red', markersize=5, 
                 markeredgewidth=markerEdgeWidth)
        ax2.plot(RSX90Points[1:,1], ZSX90Points[1:,1], marker, color='blue', markersize=markerSize, 
                 markeredgewidth=markerEdgeWidth)
        ax0.plot(RDISRADPoints[1:7,1], ZDISRADPoints[1:7,1], marker, color='gray', markersize=markerSize, 
                 markeredgewidth=markerEdgeWidth)    
        ax0.plot(RDISRADPoints[7:,1], ZDISRADPoints[7:,1], marker, color='red', markersize=markerSize, 
                 markeredgewidth=markerEdgeWidth)    
    



    
    
    
    # INCLINATION ANGLES =====================================================
    
    aveRSX90 = np.mean([axuv90Rp1UpR, axuv90Rp1LowR])
    aveZSX90 = np.mean([axuv90Rp1UpZ, axuv90Rp1LowZ])
    aveRDIS = axuv210Rp1LowR #np.mean([axuv210Rp1UpR, axuv210Rp1LowR])
    aveZDIS = axuv210Rp1LowZ #np.mean([axuv210Rp1UpZ, axuv210Rp1LowZ])
    
    
    thetaIncSX90 = np.arctan2(aveRSX90 - RSX90Points, aveZSX90 - ZSX90Points)*180./np.pi
    thetaIncDISRAD = np.arctan2(aveRDIS - RDISRADPoints, aveZDIS - ZDISRADPoints)*180./np.pi
    

    time = np.linspace(0,2.5, num=numPts -1)

    
    if i ==1:
        ax1.plot(time, thetaIncSX90[1:,0], '-x', markersize=5,color='blue')       
        ax1.plot(time, thetaIncDISRAD[1:,0], '-x', markersize=5,color='red')  
        ax1.plot(time[0:7], thetaIncDISRAD[1:8,1], '-x', markersize=3,color='gray')
        ax1.plot(time[6:], thetaIncDISRAD[7:,1], '-x', markersize=3,color='red')          
        ax1.plot([0, 3], [thetaIncSX90[1,0],thetaIncSX90[1,0]], '--', color='blue')   
        ax1.plot([0, 3], [thetaIncDISRAD[1,0], thetaIncDISRAD[1,0]], '--', color='red')
        
        np.savetxt('/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Nominal_Traj/CoIp/thetaIncDisradCo.txt', 
                   thetaIncDISRAD[1:])
        np.savetxt('/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Nominal_Traj/CoIp/thetaIncSX90Co.txt', 
                   thetaIncSX90[1:])
        np.savetxt('/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Nominal_Traj/CoIp/thetaIncTime.txt', 
                   time)       
    else:
        ax1.plot(time, thetaIncSX90[1:,0], '-o', markersize=3,color='blue')       
        ax1.plot(time, thetaIncDISRAD[1:,0], '-o', markersize=3,color='red')   
        ax1.plot(time, thetaIncSX90[1:,1], '-o', markersize=1,color='blue')       
        ax1.plot(time, thetaIncDISRAD[1:,1], '-o', markersize=1,color='red')         
        ax1.plot([0, 3], [thetaIncSX90[1,0],thetaIncSX90[1,0]], '--', color='blue')   
        ax1.plot([0, 3], [thetaIncDISRAD[1,0], thetaIncDISRAD[1,0]], '--', color='red')    
        
        np.savetxt('/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Nominal_Traj/CtIp/thetaIncDisradCt.txt', 
                   thetaIncDISRAD[1:])
        np.savetxt('/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Nominal_Traj/CtIp/thetaIncSX90Ct.txt', 
                   thetaIncSX90[1:])
        np.savetxt('/Users/Ryan/Google_Drive/ITER_Laptop/Work/DIIID_Experiment/Raman/TRIP3D/172281/SPI_Trajectory/Nominal_Traj/CtIp/thetaIncTime.txt', 
                   time)         


ax2.plot(RBdry, ZBdry, color='black')
ax0.plot(RBdry, ZBdry, color='black')    

ax0.text(1.18, 0.83, 'Time', rotation=260)
ax0.arrow(1.30, 0.45, 0.03, -0.1, head_width=0.1)
ax0.arrow(2.12, -0.3, -0.02, 0.1, head_width=0.1)    

ax2.arrow(1.75, 0.8, -0.03, -0.2, head_width=0.1)
ax2.arrow(1.49, -1, -0.02, 0.1, head_width=0.1)    


# saving theta_inc time traces to file for plotting in IDL

