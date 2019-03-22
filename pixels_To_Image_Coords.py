# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 09:25:19 2018

@author: sweener
"""

def pixels_To_Image_Coords(XPix, YPix, TotalXPixels, TotalYPixels, ScaledWidth, ScaledHeight, HorizOffset, VertOffset):
    '''
    Given the pixel coordinates of an image, the total pixel dimensions, and 
    the scale width and horizontal offset of the new image, convert the given
    pixels to the new coordinates.
    '''
    
    newX = XPix/TotalXPixels*ScaledWidth + HorizOffset - ScaledWidth/2.
    
    # we must reverse the direction of the y-coordinate
    newY = (TotalYPixels - YPix)/TotalYPixels*ScaledHeight + VertOffset - ScaledWidth/2.
    
    return newX, newY