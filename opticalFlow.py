import cv2 as cv
import numpy as np

def opticalFlow(frame,prev_frame, mask,enable_manualFilter,of_filterThreshold):


    flow = cv.calcOpticalFlowFarneback(prev_frame, frame,
                                           None,
                                           0.5, 3, 15, 3, 5, 1.2, 0)

    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Filter according to threshold -> the effort is to remove static noise from the camera
    if (enable_manualFilter== True):
        magnitude[magnitude<of_filterThreshold] = 0;

    #print(np.amax(magnitude))


    # Sets image hue according to the optical flow
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2

    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        #gb_rgb = cv.cvtColor(gb_mask, cv.COLOR_HSV2BGR)

    return(rgb)
