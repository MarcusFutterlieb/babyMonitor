import cv2 as cv
import time
import numpy as np
from opticalFlow import opticalFlow
from imagePreprocessing import imagePreProcessing
from hearRateMonitor import heartRateMonitor
from supportFunctions import getImageCenterCrop
from matplotlib import pyplot as plt

###################################
#######user setup##################
enable_gb = False # enables gaussian blur
enable_fdn = False # enables fast de-noising
enable_manualFilter = False # enables magnitude cutoff for optical flow
of_filterThreshold = 1.0 # threshold for magnitude in optical flow
enable_fd = True # enables face detection --> this might not work very well for an infant
enable_hrd = True # enables heart rate monitoring using rgb image
###################################





cv.namedWindow("cam")
videoCapture = cv.VideoCapture(0)

if videoCapture.isOpened(): # try to get the first frame
    rval, frame = videoCapture.read()
else:
    rval = False

# load cascade if fd is enabled
if (enable_fd==True):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

# Converts each frame to grayscale - we previously
# only converted the first frame to grayscale
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
prev_gray = gray

# Creates an image filled with zero
# intensities with the same dimensions
# as the frame
mask = np.zeros_like(frame)

# Sets image saturation to maximum
mask[..., 1] = 255

if(enable_hrd==True):
    x, y, w, h = getImageCenterCrop(videoCapture.get(cv.CAP_PROP_FRAME_WIDTH), videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT))
    heartbeat_count = 128
    heartbeat_values = [0] * heartbeat_count
    heartbeat_times = [time.time()] * heartbeat_count  # Matplotlib graph surface
    fig = plt.figure()
    ax = fig.add_subplot(111)




while rval:
    # create grayscale image from camera frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detect face if face detection is enabled by the user
    if (enable_fd == True):
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)



    cv.imshow("cam", frame)


    gray = imagePreProcessing(gray,enable_gb,enable_fdn)



    cv.imshow("input", gray)
    rgb = opticalFlow(gray,prev_gray, mask,enable_manualFilter,of_filterThreshold)
    prev_gray = gray
    cv.imshow("dense optical flow", rgb)

    # Get a new frame from the camera feed
    rval, frame = videoCapture.read()

    # Get heartrate from red channel
    if (enable_hrd == True):
        heartbeat_values, heartbeat_times = heartRateMonitor(frame, x, y, w, h, heartbeat_values, heartbeat_times, ax, fig)

    # Frames are read by intervals of 20 millisecond. The
    # programs breaks out of the while loop when the
    # user presses the 'q' key
    key = cv.waitKey(20)
    if (key & 0xFF == ord('q')) or (key == 27):  # exit on ESC and q:
        break

videoCapture.release()
cv.destroyWindow("preview")