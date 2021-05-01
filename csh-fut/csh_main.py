import numpy as np
import cv2
from csh import csh

patch_size_standard = 8
debug = 1
c = csh()

# Takes image in BGR format(standard cv2 format)
# returns Y Cr Cb image
def convert_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB).astype(np.uint8)

# takes matches as shape of (x,y,2)
# takes original and target image as BGR format
# takes patch_size as int
# returns error as the average l2 distance
def mean_error(matches, orig_img, trg_img, patch_size=patch_size_standard):
    err = c.RMS_error(orig_img, trg_img, matches)
    return err

    # errs = np.zeros_like(matches[:,:,0])
    # x,y,z = orig_img.shape
    # xs, ys = matches[:,:,0], matches[:,:,1]
    # for i in np.arange(0,patch_size):
    #     for j in np.arange(0,patch_size):
    #         errs+= np.sum((orig_img[i:x+i-patch_size+1, j:y+j-patch_size+1,:]-trg_img[xs+i,ys+j,:])**2,axis=2) 
    # return np.mean(errs**0.5)

# takes matches as shape of (x,y,k,2)
# takes original and target image as BGR format
# takes patch_size as int
# returns matches as (x,y,2)
# Computes the actual distances and per patch returns the match with lowest distance.
def find_best_n(matches, orig_img, trg_img, patch_size=patch_size_standard):
    patches = c.pick_best_nn(orig_img, trg_img, matches).get()
    return patches
    #Trash implementation of picking the best. Took ~20x as long as finding the patches. Moved to Futhark for a ~400x speedup
    # dist = np.zeros_like(matches[:,:,:,0])
    # x,y,z = orig_img.shape
    # xs, ys = matches[:,:,:,0], matches[:,:,:,1]
    # for i in np.arange(0,patch_size):
    #     for j in np.arange(0,patch_size):
    #         dist += np.sum((orig_img[i:x+i-patch_size+1, j:y+j-patch_size+1,None,:]-trg_img[xs+i,ys+j,:])**2,axis=3)
    # best_inds = np.argmin(dist, axis=2)
    # return np.take_along_axis(matches,best_inds[:,:,None,None], axis=2).squeeze()

# takes original and target image as Y Cb Cr format
# takes patch_size as int, 8 is currently the only option
# takes iterations as int, this is how many times the propagation runs
# Takes knn as int, this is how many nearest neighbours are found.
# Returns knn for each patch.
def csh_knn(orig_img, trg_img, iters, knn, patch_size=patch_size_standard):
    knn_patches = c.main(orig_img, trg_img, iters, knn).get()
    return knn_patches

def main(orig_img_path, trg_img_path, iters=5, knn=3, patch_size=patch_size_standard):
    img_a = cv2.imread(orig_img_path)
    img_b = cv2.imread(trg_img_path)
    img_a_conv = convert_image(img_a)
    img_b_conv = convert_image(img_b)
    if debug:
        print("Original image shape: {}".format(img_a_conv.shape))
        print("Target image shape: {}".format(img_b_conv.shape))
    knn_patches = csh_knn(img_a_conv, img_b_conv, iters, knn)
    img_a_u8 = img_a.astype(np.uint8)
    img_b_u8 = img_b.astype(np.uint8)
    best_matches = find_best_n(knn_patches, img_a_u8, img_b_u8)
    # img_a = img_a.astype(int)
    # img_b = img_b.astype(int)
    return mean_error(best_matches, img_a_u8, img_b_u8)