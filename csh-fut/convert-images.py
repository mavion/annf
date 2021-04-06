import futhark_data
import cv2
import numpy as np
image_a = cv2.imread("../csh-naive/data/example/1.jpg")
image_b = cv2.imread("../csh-naive/data/example/2.jpg")
image_a2 = cv2.cvtColor(image_a, cv2.COLOR_BGR2YCR_CB).astype(np.int8)
image_b2 = cv2.cvtColor(image_b, cv2.COLOR_BGR2YCR_CB).astype(np.int8)
with open('imgs.npy', 'wb') as f:
    futhark_data.dump(np.moveaxis(image_a2,2,0), f)
    futhark_data.dump(np.moveaxis(image_b2,2,0), f)