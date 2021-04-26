import "whprojections"
import "hashing"
import "propagate"

-- Given two images in Y/Cr/Cb format return the annf. Hardcoded to 8x8 patch size for now
let cshANN [n] [m] [k] [j]
           (img_src : [n][m][3]u8) 
           (img_trg : [k][j][3]u8) 
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
    let bit_counts = [5, 3, 3, 1, 1, 1, 2, 2] -- how many bits are allocated to specific kernels, unmentioned kernels are 0 and thus skipped
    let kernels_used = [0, 1, 5, 6, 2, 9, 15, 19] -- kernels used for creating hashcodes, implicit indexing used for the number of bits per kernel
    let (hash_src, hash_trg) = create_hash_codes (map (\i -> wh_src[i,:]) kernels_used) (map (\i -> wh_trg[i,:]) kernels_used) iters bit_counts
    -- create hashtables --
    let width = 2
    let hash_table_src = create_hash_table hash_src (1<<(reduce (+) 0 bit_counts)) width
    let hash_table_trg = create_hash_table hash_trg (1<<(reduce (+) 0 bit_counts)) width
    -- searching --
    -- initialize matches --
    let wh_src_trs = transpose wh_src
    let wh_trg_trs = transpose wh_trg
    let (matches,matchl2) = init_matches wh_src wh_trg knn
    -- propapagation --
    let (matches,_) = 
        loop (matches, matchl2) = (matches, matchl2) for i < iters do
            -- find all 3 types of candidates
            let candidates = find_candidates_all matches hash_src[i,:] hash_trg[i,:] hash_table_src[i,:] hash_table_trg[i,:] dimy dimy2
            -- find l2 distance of all candidates
            let candidatesl2 = map2 (\x ys -> map (\y -> dist2 wh_src_trs[x,:] wh_trg_trs[y,:]) ys) (iota patch_count) candidates
            -- pick the knn best candidates from candidates and matches. These are the new matches
            let (matches', matchl2') = unzip (map4 (pick_best) matches matchl2 candidates candidatesl2)
            in (matches', matchl2')
    -- convert from 1d coordinates to 2d
    in unflatten dimx dimy (map (\xs -> map (\x -> [x / dimy, x % dimy]) xs) matches)


-- Given two images in RGB format(or similar, fx RBG) and the knn produced, return the nn.
entry pick_best_nn [n] [m] [k] [j] [knn] [r] [s]
                (img_src: [n][m][3]u8)
                (img_trg: [k][j][3]u8)
                (matches: [r][s][knn][2]i64)
                : [r][s][2]i64 =
    let patch_size = 8*8*3
    in map2 (\x match_row ->  
        map2 (\y match_point ->
            let src_patch = map (i64.u8) (flatten_3d img_src[x:x+8,y:y+8,:]) :> [patch_size]i64
            let dists = map (\xy -> dist2 src_patch (map (i64.u8) (flatten_3d img_trg[xy[0]:xy[0]+8,xy[1]:xy[1]+8,:]) :> [patch_size]i64)) match_point
            let (_, best) = reduce (\(x0, x1) (y0, y1) -> if x0 < y0 then (x0,x1) else (y0,y1)) (i64.highest, [0,0]) (zip dists match_point)
            in best
        ) (iota s) match_row
    ) (iota r) matches

-- Given two images in RGB format(or similar, fx RBG) and the nn produced, return the RMS
-- Why does the below give an allocation error when the above doesn't?
-- entry RMS_error [n] [m] [k] [j] [r] [s]
--                 (img_src: [n][m][3]i64)
--                 (img_trg: [k][j][3]i64)
--                 (matches: [r][s][2]i64)
--                 : f32 =
--     let patch_size = 8*8*3
--     let l2s =  
--         map2 (\x match_row ->  
--             map2 (\y nn ->
--                 let src_patch = flatten_3d img_src[x:x+8,y:y+8,:] :> [patch_size]i64
--                 let trg_patch = flatten_3d img_trg[nn[0]:nn[0]+8,nn[1]:nn[1]+8,:] :> [patch_size]i64
--                 in dist2 src_patch trg_patch
--             ) (iota s) match_row
--         ) (iota r) matches
--     let l2s = map (f32.i64) (flatten l2s)
--     let patch_count = f32.i64 (r*s)
--     let l2s = map (/patch_count) l2s
--     in (reduce (+) 0 l2s)**0.5


let main img_a img_b iters knn = cshANN img_a img_b iters knn