import cv2
import numpy as np
from numba import jit
import keyboard
import time


def norm_cv(image, a, b):
    img = np.zeros((800, 800))
    norm_image = cv2.normalize(image, img, a, b, cv2.NORM_MINMAX)
    return norm_image

#normalization with numpy
def norm_python(image, a, b):
    norm_image = (a + (image - image.min())/(image.max() - image.min()) * (b - a)).astype(np.uint8)
    norm_image = np.round(norm_image)
    return norm_image

#normalization with numpy + numba
@jit(forceobj=True)
def norm_jit(image, a, b):
    norm_image = (a + (image - image.min())/(image.max() - image.min()) * (b - a)).astype(np.uint8)
    norm_image = np.round(norm_image)
    return norm_image

image = cv2.imread('Field.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800, 500)
cv2.imshow('image', gray)
cv2.waitKey(0)
while True:
    if keyboard.is_pressed('z'):
        start = time.perf_counter()
        # norm = norm_python(gray, 100, 150)    #Elapsed = 0.030973799992352724s
        norm = norm_jit(gray, 100, 150)       #Elapsed = 0.2882584000035422s
                                            #Elapsed = 0.032154299988178536s
        # norm = norm_cv(gray, 100, 150)  #Elapsed = 0.001137600003858097s
        end = time.perf_counter()
        print("Elapsed = {}s".format((end - start)))
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 800, 500)
        cv2.imshow('image', norm)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if keyboard.is_pressed('x'):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 800, 500)
        cv2.imshow('image', gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if keyboard.is_pressed('1'):
        cv2.destroyAllWindows()
        break
