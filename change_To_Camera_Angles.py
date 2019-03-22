# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:02:46 2018

@author: sweener
"""

import numpy as np

def change_To_Camera_Angles(X,Y,Z, Sigc, Gamma, Azim, Rc, phiMach, Zc):
    
    '''
    Given a point in machine X,Y,Z coordinates, convert this to angles
    alpha and beta in the camera view plane. Sigc is the angle of the camera
    in the XY plane, and Gamma is the angle of the camera in the below the 
    horizontal plane. 
    '''
    
    deg2Rad = np.pi/180.
    
    #Location of the camera. For the moment, we assume outboard midplane
    # at phi=0
    
    #Rc = 2.537
    #phiMach = 226.8
    Xc = Rc*np.cos((360.-phiMach)*deg2Rad) # m
    Yc = Rc*np.sin((360.-phiMach)*deg2Rad) # m
    #Zc = -0.140 # m
    
    
    camFieldView =35. # deg

    
    # move the origin to the origin of the camera
    Xtemp = X - Xc
    Ytemp = Y - Yc
    Ztemp = Z - Zc
    
    # for the moment we assume the view chord is in a Z=const. plane, but 
    # we will allow here for a rotation in the X,Y plane
    Xtemp2 = Xtemp*np.cos(Sigc*deg2Rad) + Ytemp*np.sin(Sigc*deg2Rad)
    Ytemp2 = Ytemp*np.cos(Sigc*deg2Rad) - Xtemp*np.sin(Sigc*deg2Rad)
    Ztemp2 = Ztemp
    
    # we now need to allow for the camera to look off the z=const. plane. this
    # requires a rotation in the xz plane. 
    Xtemp3 = Xtemp2*np.cos(Gamma*deg2Rad) + Ztemp2*np.sin(Gamma*deg2Rad)
    Ztemp3 = Ztemp2*np.cos(Gamma*deg2Rad) - Xtemp2*np.sin(Gamma*deg2Rad)
    Ytemp3 = Ytemp2
    
    # finally, we will allow for a rotation of the camera along the view chord
    Xtemp4 = Xtemp3
    Ytemp4 = Ytemp3*np.cos(Azim*deg2Rad) + Ztemp3*np.sin(Azim*deg2Rad)
    Ztemp4 = Ztemp3*np.cos(Azim*deg2Rad) - Ytemp3*np.sin(Azim*deg2Rad)
    
    # the camera view chord is now looking down the Xtemp2-axis, in the -x 
    # direction. The image plane is thus the yz plane. Alpha is now our angle
    # that effectively functions like a radius, and beta is our azimuthal coordinate.
    # By changing y and z into alpha and beta, we can now fully recontruct the 
    # camera image. Alpha = 0 means the view chord is parallel to the x-axis. 
    # Beta = 0 means that the point lies in the z=0 plane. 
    
    # lets get rid of all data points with positive x
    frontCamera = np.where(Xtemp2 < 0.)
    
    Xtemp4 = Xtemp4[frontCamera]
    Ytemp4 = Ytemp4[frontCamera]
    Ztemp4 = Ztemp4[frontCamera]
    
    alpha = np.arctan(np.sqrt(Ztemp4**2 + Ytemp4**2)/np.abs(Xtemp4))
    beta = np.arctan2(Ztemp4, Ytemp4)
    
    # Let's restrict the field of view
    fieldOfView = np.where(alpha < camFieldView*deg2Rad)
    alpha2 = alpha[fieldOfView]
    beta2 = beta[fieldOfView]
    
    # We are now ready to map into our 2D camera image. Alpha here, though an 
    # angle, is equivalently our radial coordinate. 
    rIm = alpha2
    thetaIm = beta2
    
    xIm = rIm*np.cos(thetaIm)
    yIm = rIm*np.sin(thetaIm)
    
    return xIm, yIm