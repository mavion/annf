import "propagate"

-- Demonstrating the autotune bug
-- ==
-- entry: main
-- compiled input @ auto_bug.in

entry main [iters] [patch_count] [n] [k]
    (hash: [iters][patch_count]i64)
    (hash_table: [iters][n][2]i64)
    (wh: [patch_count][k]i64)
    (knn: i64)
    (dimy: i64)
    (dimy2: i64)
    =
    let (matches,matchl2) = init_matches (transpose wh) (transpose wh) knn
    let (matches,_) = 
        loop (matches, matchl2) = (matches, matchl2) for i < iters do
            -- find all 3 types of candidates
            let candidates = find_candidates_all matches hash[i,:] hash[i,:] hash_table[i,:] hash_table[i,:] dimy dimy2
            -- find l2 distance of all candidates
            let candidatesl2 = map2 (\x ys -> map (\y -> dist2 wh[x,:] wh[y,:]) ys) (iota patch_count) candidates
            -- let candidatesl2 = dist2_all wh_src_trs wh_trg_trs candidates
            -- pick the knn best candidates from candidates and matches. These are the new matches
            let (matches', matchl2') = unzip (map4 (bruteForcePar) matches matchl2 candidates candidatesl2)
            in (matches', matchl2')
    in matches