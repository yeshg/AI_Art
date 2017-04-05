import numpy as np
import cv2
img = cv2.imread("GoogLeNet_deepdraw_denoised_79.jpg")

def nothing(*arg):
    pass



# Bilateral Filtering - removing noise - cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]])
# http://people.csail.mit.edu/sparis/bf_course/slides/03_definition_bf.pdf

# Size needed for sigmaSpace = 2% of img diagonal
height, width, channels = img.shape
diag = np.sqrt(width**2 + height**2)
sigmaSpace = 0.02 * diag
sigmaColor = 40

# Make Window and Trackbar
cv2.namedWindow('Cartooned Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Cartooned Image', 2003,2003)
cv2.createTrackbar('Cartooned Image sigmaSpace','Cartooned Image',0,50,nothing)
cv2.createTrackbar('Cartooned Image sigmaColor','Cartooned Image',0,50,nothing)

# Allocate Destination Image
denoisedImage = np.zeros(img.shape,np.uint8)

# Loop for get trackbar pos and process it
while True:
    # Get trackbar pos
    TrackbarPos1 = cv2.getTrackbarPos('Cartooned Image sigmaSpace','Cartooned Image')
    TrackbarPos2 = cv2.getTrackbarPos('Cartooned Image sigmaColor','Cartooned Image')
    # Apply Bilateral Filter
    cv2.bilateralFilter(img, -1, TrackbarPos1, TrackbarPos2, denoisedImage) # values have been tuned, no trackbar adjustments will change anything
    # Show in Window
    cv2.imshow('Cartooned Image', denoisedImage)

    # If you press "ESC", it will return value
    ch = cv2.waitKey(50)
    if ch == 27:
        break

cv2.waitKey(0)
