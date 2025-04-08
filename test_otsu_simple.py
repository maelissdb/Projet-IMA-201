# Test file to display segmentation with Otsu's method without pre or post post-processing
from display_image import mask_display, viewimage
from otsu_seg import otsu
import cv2
from otsu2 import display_otsu_level

img = cv2.imread("images_test/img10.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def display_otsu_simple(img):
    tresh = otsu(img,None)
    new_img = mask_display(img,None,tresh)
    return new_img
