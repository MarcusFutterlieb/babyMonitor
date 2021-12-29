# here we have short supporting functions with no fixed topic
import cv2 as cv

def getImageCenterCrop(resolutionX,resolutionY):
    w = 100
    h = 100
    x = (resolutionX-w)/2
    y = (resolutionY-h)/2
    x = int(x)
    y = int(y)
    h = int(h)
    w = int(w)

    return(x,y,w,h)