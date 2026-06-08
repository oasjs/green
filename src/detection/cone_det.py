# Main References:
# Libs:
# https://docs.opencv.org/3.4/ for OpenCV documentation - main library of the project
# https://numpy.org/doc/ for NumPy documentation - main library for matrix/vectorial calculations
# https://www.geeksforgeeks.org/computer-vision/image-processing-algorithms-in-computer-vision/ for getting the content seen during classes summarized
# https://docs.opencv.org/3.4/dd/d53/tutorial_py_depthmap.html for understanding depth map (suits better the other part of the project);
# https://www.geeksforgeeks.org/computer-vision/comprehensive-guide-to-edge-detection-algorithms/ for possible edge detection in geometry matching on the cone;
# https://www.geeksforgeeks.org/computer-graphics/hsv-color-model-in-computer-graphics/ for understanding HSV color model
# https://www.geeksforgeeks.org/python/filter-color-with-opencv/ for HSV filters/masks;
# https://github.com/computervisionpro/yt/blob/main/hsv_masking/color-nemo.py for hints to defining th range of the HSV;
# https://www.geeksforgeeks.org/python/clahe-histogram-eqalization-opencv/ for CLAHE in brightness equalisation;
# MOD 1: Cap. 1.2.1 - morphological operations for eliminating unwanted noise and aggregating separated pieces;
# MOD 1: Cap. 1.2.2 - for identifying the borders of the cones and, therefore their geometry;
# MOD 1: Cap 1.3.1 - for normalizing the HSV filter;

# Secondary Refereces:
# https://revisitingmiwb.github.io/ for understanding white balancing;

from __future__ import annotations
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

# Yellow
YELLOW_LOW  = np.array([ 18, 120, 80])
YELLOW_HIGH = np.array([ 35, 255, 255])

# Blue
BLUE_LOW    = np.array([ 95, 120, 50])
BLUE_HIGH   = np.array([135, 255, 255])

# Orange
ORANGE_LOW  = np.array([  5, 150, 80])
ORANGE_HIGH = np.array([ 18, 255, 255])

PATH_TO_IMAGES = '../../assets/images/'

# The HSV filters/masks must account for both colors present in the target cone:
# YELLOW: yellow and black; BLUE: blue and white
def apply_color_mask(img, hsv, color):
    # Define range for blue color in HSV
    # It is here that the biggest change may occur
    # Caution with this section
    # The conditions are not ideal -> shall be changed
    if color == 0:
        lower = BLUE_LOW
        upper = BLUE_HIGH
    elif color == 1:
        lower = YELLOW_LOW
        upper = YELLOW_HIGH
    elif color == 2:
        lower = ORANGE_LOW
        upper = ORANGE_HIGH

    # Create mask
    mask = cv2.inRange(hsv, lower, upper)

    # Filter the blue region
    result = cv2.bitwise_and(img, img, mask=mask)

    # Show imgs
    cv2.imshow('Original img', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('Filtered Result', result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# For white balancing according to
def percentile_stretch(img_bgr, percentile=95):
    img = img_bgr.astype(np.float32)
    for i in range(3):
        p = np.percentile(img[:,:,i], percentile)
        if p > 0:
            img[:,:,i] = img[:,:,i] * (255.0 / p)
    return np.clip(img, 0, 255).astype(np.uint8)

# For reducing noise - undesired bits extracted from applying the mask
def apply_morphological_cleaning(img):
    # img var must be in RGB to be converted as shown below
    img_bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.erode(img_bw)
    cv2.dilate(img_bw)
    
# # For recognizing the cone geometry
# # TODO: Develop the algorithm and apply the contour filters using cv2 -> a source was not searched yet.
# def apply_contour_detection(img):
    

# https://www.geeksforgeeks.org/python/clahe-histogram-eqalization-opencv/
# For adjusting only brightness in the context of using HSV
# Observe that only the Value V receives CLAHE processing
def apply_clahe_contrast(hsv, clipLimit=2.0, tileGridSize=(8,8)):
    # image_bw = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    hsv[:, :, 2] = np.clip(clahe.apply(hsv[:, :, 2]) + 30, 0, 255).astype(np.uint8)
    # _, threshold_img = cv2.threshold(hsv, 155, 255, cv2.THRESH_BINARY)
    # display_image("Ordinary Threshold", threshold_img)
    display_image("CLAHE Image", hsv)
    
    return hsv

# Displays the image
def display_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# For plotting HSV in a cartesian space
def plot(img_rgb, hsv):
    pixel_colors = img_rgb.reshape((np.shape(img_rgb)[0]*np.shape(img_rgb)[1], 3))
    print(pixel_colors)
    print(pixel_colors.shape)
    print()

    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    h, s, v = cv2.split(hsv)
    hf, sf, vf = h.flatten(), s.flatten(), v.flatten()
    print(hf)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(hf, sf, vf, facecolors=pixel_colors, marker=".")

    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")

    plt.show()

# Apply white balancing, HSV conversion and clahe contrast as preprocessing steps
def preprocess(img) -> cv2.typing.MatLike:
    # White Balance WB
    img = percentile_stretch(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = apply_clahe_contrast(img)
    return hsv

def main():
    # Read the entirety of the image passed through its path
    # TODO: generalize the image path to make the modifictions easier
    img = cv2.imread(PATH_TO_IMAGES + '000004.png')

    # Shows the original image
    cv2.imshow("ORIGINAL IMAGE:", img)

    # Print width, length and channels (the latter for each pixel)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img_rgb)

    # Convert BGR to HSV
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = preprocess(img)

    # Prints a complete plot for viewing relevant information
    plot(img_rgb, hsv)

    # Loops for color in colors
    # Currently: [YELLOW, BLUE]
    colors = [0, 1]
    for color in colors:
        # Calls function to apply mask color according to the color
        apply_color_mask(img, hsv, color)
    

    print("Exiting...")  # Confirm exit

if __name__ == "__main__":
    main()
