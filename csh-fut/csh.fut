import "whprojections"
import "hashing"
import "propagate"

-- Given two images in Y/Cr/Cb format return the annf. Hardcoded to 8x8 patch size for now
let cshANN [n] [m] [k] [j]
           (img_src : [3][n][m]i8) 
           (img_trg : [3][k][j]i8) 
           (iters: i64)
           (knn: i64)
           : [][][knn][2]i64 =
    -- indexing --
    -- compute projections --
    let (dimx, dimy) = (n-7, m-7)
    let patch_count = dimx*dimy
    let wh_src = wh_project_8bit img_src :> [23][patch_count]i64
    let (dimx2, dimy2) = (k-7, j-7)
    let patch_count2 = dimx2*dimy2
    let wh_trg = wh_project_8bit img_trg :> [23][patch_count2]i64
    -- create hashcodes --
    let bit_counts = [5, 3, 3, 1, 1, 1, 2, 2] -- how many bits are allocated to specific kernels, unmentioned kernels are 0
    let kernels_used = [0, 1, 5, 6, 2, 9, 15, 19] -- kernels used for creating hashcodes, implicit indexing used for the number of bits per kernel
    let (hash_src, hash_trg) = create_hash_codes (map (\i -> wh_src[i,:]) kernels_used) (map (\i -> wh_trg[i,:]) kernels_used) iters bit_counts
    -- create hashtables --
    let width = 2
    let hash_table_src = create_hash_table hash_src (1<<(reduce (+) 0 bit_counts)) width
    let hash_table_trg = create_hash_table hash_trg (1<<(reduce (+) 0 bit_counts)) width
    -- searching --
    -- initialize matches --
    let (matches,matchl2) = init_matches wh_src wh_trg knn
    -- propapagation --
    let (matches,_) = 
        loop (matches, matchl2) = (matches, matchl2) for i < iters do
            -- find all 3 types of candidates
            let candidates = find_candidates_all matches hash_src[i,:] hash_trg[i,:] hash_table_src[i,:] hash_table_trg[i,:] dimy dimy2
            -- find l2 distance of all candidates
            let candidatesl2 = map2 (\x ys -> map (\y -> dist2 wh_src[:,x] wh_trg[:,y]) ys) (iota patch_count) candidates
            -- pick the knn best candidates from candidates and matches. These are the new matches
            let (matches', matchl2') = unzip (map4 (pick_best) matches matchl2 candidates candidatesl2)
            in (matches', matchl2')
    -- convert from 1d coordinates to 2d
    in unflatten dimx dimy (map (\xs -> map (\x -> [x % dimx, x / dimx]) xs) matches)


let main img_a img_b = cshANN img_a img_b 5 2