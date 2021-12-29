import cv2 as cv
from gaussianBlur import gaussianBlur

def imagePreProcessing (frame,enable_gb,enable_fdn):
    if (enable_fdn == True):
        frame = cv.fastNlMeansDenoising(frame)
    if (enable_gb == True) :
        frame = gaussianBlur(frame)

    return(frame)