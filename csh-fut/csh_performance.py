import sys
import os
import cv2
import time
import numpy as np
import csh_main

debug=1
limit = 5
stepsize_k = 2
stepsize_iterations = 2
max_k = 8
max_iterations = 10
max_pairs = 5

directory_path = sys.argv[1]
entries = sorted(os.listdir(directory_path))

result_file = open('results.data', 'w')

def run_pair(path1, path2):
    if debug: print("Image name: {}".format(path1))
    time1 = time.perf_counter()
    img_a = cv2.imread(path1)
    img_b = cv2.imread(path2)
    img_a_conv = csh_main.convert_image(img_a)
    img_b_conv = csh_main.convert_image(img_b)
    img_a_u8 = img_a.astype(np.uint8)
    img_b_u8 = img_b.astype(np.uint8)
    img_a = img_a.astype(int)
    img_b = img_b.astype(int)
    time2 = time.perf_counter()
    result_file.write("{},{},{}\n".format(path1, path2, time2-time1))
    for j in range(1,max_k, stepsize_k):
        for i in range(1,max_iterations, stepsize_iterations):
            if debug: print("Iterations {}, Knn {}".format(i,j))
            time1 = time.perf_counter()
            res = csh_main.csh_knn(img_a_conv, img_b_conv, i,j)
            time2 = time.perf_counter()
            best_matches = csh_main.find_best_n(res, img_a_u8, img_b_u8)
            time3 = time.perf_counter()
            error = csh_main.mean_error(best_matches, img_a, img_b)
            result_file.write("{},{},{}\n".format(time2-time1, time3-time2, error))

pair_count = 0
for i,k in zip(entries[0::2],entries[1::2]):
    run_pair(os.path.join(directory_path, i),(os.path.join(directory_path,k)))
    pair_count +=1
    if pair_count >= max_pairs: break

result_file.close()
    