import numpy as np
import cv2
from matplotlib import pyplot as plt


def template_match(image_og, image_temp):
    w, h = image_temp.shape[::-1]
    res = cv2.matchTemplate(image_og, image_temp,cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image_og,top_left, bottom_right, 255, 2)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 800, 500)
    cv2.imshow('image', image_og)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sift_bf(image_og, image_temp):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image_og, None)
    kp2, des2 = sift.detectAndCompute(image_temp, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(image_og, kp1, image_temp, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()

def sift_flann(image_og, image_temp):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image_og, None)
    kp2, des2 = sift.detectAndCompute(image_temp, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params,search_params)
    knn_matches = matcher.knnMatch(des1, des2, 2)

    ratio_thresh = 0.6
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    img_matches = np.empty((max(image_og.shape[0], image_temp.shape[0]), image_og.shape[1] + image_temp.shape[1], 3),
                           dtype=np.uint8)
    img3 = cv2.drawMatches(image_og, kp1, image_temp, kp2, good_matches[:10], img_matches,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()


image_og = cv2.imread('Eifel.jpg', 0)
image_test = cv2.imread('Eifel4.jpg', 0)
# template_match(image_og, image_test)
sift_bf(image_og,image_test)
sift_flann(image_og,image_test)
