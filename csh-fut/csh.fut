import "whprojections"
import "hashing"
import "propagate"

-- Given two images in Y/Cr/Cb format return the annf. Hardcoded to 8x8 patch size for now
let cshANN [n] [m] [k] [j]
           (img_src : [3][n][m]i8) 
           (img_trg : [3][k][j]i8) 
           (iters: i64)
           : []i64 =
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
    let matches = init_matches patch_count patch_count2
    let matchl2 = map2 (\x y -> dist2 wh_src[:,x] wh_trg[:,y]) (iota patch_count) matches
    -- propapagation --
    let (matches,_) = 
        loop (matches, matchl2) = (matches, matchl2) for i < iters do
            -- find all 3 types of candidates
            let candidates = find_candidates matches hash_src[i,:] hash_trg[i,:] hash_table_src[i,:] hash_table_trg[i,:] (dimx, dimy)
            -- compute l2 dist(lower bound) for each candidate and find best candidate
            let best_cand = best_dist wh_src wh_trg candidates
            -- compare best candidate and match and return best of the two
            let (matches', matchl2') = unzip (cmp_cand_match (zip matches matchl2) best_cand)
            in (matches', matchl2')
    -- convert from 1d coordinates to 2d
    in matches

let main img_a img_b = cshANN img_a img_b 5