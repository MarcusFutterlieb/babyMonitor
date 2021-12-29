# guide lines taken from
# https://blog.devgenius.io/remote-heart-rate-detection-using-webcam-and-50-lines-of-code-2326f6431149
#

import numpy as np
from matplotlib import pyplot as plt
import cv2
import io
import time




def heartRateMonitor(frame,x,y,w,h,heartbeat_values,heartbeat_times,ax,fig):
    # Camera stream
    #cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
    #cap.set(cv2.CAP_PROP_FPS, 30)
    # Video stream (optional, not tested)
    # cap = cv2.VideoCapture("video.mp4")
    # Image crop (resolution of webcam is 640x480)
    #x, y, w, h = 950, 300, 100, 100


    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop_img = img[y:y + h, x:x + w]
    # Update the data
    heartbeat_values = heartbeat_values[1:] + [np.average(crop_img)]
    heartbeat_times = heartbeat_times[1:] + [time.time()]
    # Draw matplotlib graph to numpy array
    ax.plot(heartbeat_times, heartbeat_values)
    fig.canvas.draw()
    plot_img_np = np.fromstring(fig.canvas.tostring_rgb(),
                                dtype=np.uint8, sep='')
    plot_img_np = plot_img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.cla()
    # Display the frames
    cv2.imshow('Crop', crop_img)
    cv2.imshow('Graph', plot_img_np)
    return(heartbeat_values, heartbeat_times)

