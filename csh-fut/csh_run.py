import sys
import csh_main

name_a = sys.argv[1]
name_b = sys.argv[2]
iters = int(sys.argv[3])
knn = int(sys.argv[4])
patchsize = int(sys.argv[5])
err = csh_main.main(name_a, name_b, iters, knn, patchsize)
print(err)