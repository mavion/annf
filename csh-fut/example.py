import sys
import os
import cv2
import time
import numpy as np
import csh_main

#load images
image_a = cv2.imread("data/Avatar1.jpg")
image_b = cv2.imread("data/Avatar2.jpg")
#Format expected by CSH
img_a_conv = csh_main.convert_image(image_a)
img_b_conv = csh_main.convert_image(image_b)
#Format expected for errors
img_a_u8 = image_a.astype(np.uint8)
img_b_u8 = image_b.astype(np.uint8)


ps = 8
i = 20
knn = 8

time1 = time.perf_counter()
#Running KNN CSH
print("Running CSH using KNN={} for {} iterations:".format(knn,i))
res = csh_main.csh_knn(img_a_conv, img_b_conv, i,knn, patch_size=ps)
time2 = time.perf_counter()
#Finding the best match out of K
best_matches = csh_main.find_best_n(res, img_a_u8, img_b_u8,patch_size=ps)
#Finding the L2 error when given a 1NN CSH
error = csh_main.mean_error(best_matches, img_a_u8, img_b_u8, patch_size=ps)

print("Elapsed time: {} seconds.".format(time2-time1))
print("L2 Error: {}.".format(error))