import cv2 as cv

def gaussianBlur(frame):
    # apply gaussian blur to remove noise from image
    gb_kSize = [5, 5]
    gb_sigmaX = 10
    gb_sigmaY = 10
    gb_borderType = cv.BORDER_DEFAULT;
    gb_frame = cv.GaussianBlur(frame, gb_kSize, gb_sigmaX, gb_sigmaY, gb_borderType)
    return (gb_frame)