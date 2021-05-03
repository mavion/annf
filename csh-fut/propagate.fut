import "lib/github.com/diku-dk/cpprandom/random"
import "lib/github.com/diku-dk/sorts/insertion_sort"
import "helper_functions"


module rng_engine = minstd_rand
module rand_i64 = uniform_int_distribution i64 rng_engine

let dist2 [k] (xs: [k]i64) (ys: [k]i64) : i64 =
    reduce (+) 0 (map2 (\x y -> (x-y)*(x-y)) xs ys)


-- Finds the distance from all candidates
-- (same) implicit indexing of candidates and src_vals
-- uses manual flattening and segmented reduce for optimal performance
-- Tests show that this is faster than the current implementation for datasets of the expected size, however switching it in worsens performance by a significant amount.
-- presumably the nested way is easier for the compiler to fuse.
let dist2_all [n] [m] [v] [c]
    (src_vals: [n][v]i64)
    (trg_vals: [m][v]i64)
    (candidates: [n][c]i64) : [n][c]i64 =
    -- outer map is flattened
    let src_points = map (/c) (iota (n*c)) -- lazy replicated iota
    let cands_f = flatten_to (n*c) candidates
    let dists = map2 (\x y -> dist2 src_vals[x,:] trg_vals[y,:]) src_points cands_f
    in unflatten n c dists


let init_matches [k] [n] [m]
                 (wh_src: [k][n]i64) 
                 (wh_trg: [k][m]i64)
                 (knn: i64)
                 : ([n][knn]i64, [n][knn]i64) =
    let rng = rng_engine.rng_from_seed [42]
    let rngs = rng_engine.split_rng (n*knn) rng
    let (_,rands) = unzip (map (rand_i64.rand (0, m-1)) rngs)
    let matches = unflatten n knn rands
    let matchl2 = map2 (\x ys -> map (\y -> dist2 wh_src[:,x] wh_trg[:,y]) ys) (iota n) matches
    in unzip (map2 (\xs ys -> unzip (insertion_sort_by_key (\(_,y) -> y) (<=) (zip xs ys))) matches matchl2)

let find_candidates_all [n] [m] [o] [k] [knn] 
                    (matches: [n][knn]i64)
                    (hash_src: [n]i64)
                    (hash_trg: [m]i64)
                    (hash_table_src: [o][k]i64)
                    (hash_table_trg: [o][k]i64)
                    (y_size_src: i64)
                    (y_size_trg: i64)
                    : [n][]i64 =
    let cand_count = k+4*knn + 4*knn*k +k*knn
    let find_candidates i = -- type 1. check hash_src on table_trg
        let type1 = hash_table_trg[hash_src[i],:]
        -- type 2. check neighbours->match->neighbour->hash_trg on table_trg
        let neighbours = map (\step -> if i + step < 0 || i + step >= n then 0 else i + step) [1, (-1), y_size_src, (-y_size_src)]
        let match_ngbr = map (\j -> matches[j,:]) neighbours
        -- neighbours matches' neighbour are candidates
        let type2ngbr = flatten (map2 (\js step -> map (\j ->
                                    if j + step < 0 || j + step >= m then 0 else j + step) js
                                ) match_ngbr [(-1), 1, (-y_size_trg), y_size_trg])
        -- these candidates hash lookups are also candidates
        let type2hash = flatten (map (\j -> hash_table_trg[hash_trg[j],:]) type2ngbr) 
        -- type 3. check hash_src on table_src->matches
        let type3 = flatten (map (\j -> matches[j,:]) hash_table_src[hash_src[i],:])
        in type1 ++ type2ngbr ++ type2hash ++ type3 :> [cand_count]i64
    in map (find_candidates) (iota n)


let bruteForce [knn] [n]
            (matches: [knn]i64)
            (match_dist: [knn]i64)
            (candidates: [n]i64)
            (candidate_dist: [n]i64)
            : ([knn]i64, [knn]i64) =
    loop (m_inds, m_d) = (copy matches, copy match_dist) for i < n do
        if candidate_dist[i] >= m_d[knn-1] then (m_inds, m_d)
        else 
            let cur_dist = candidate_dist[i]
            let cur_ind = candidates[i]
            let (inds', il2s', _, _) =
                loop (m_inds, m_d, cur_ind, cur_dist) for j < knn do
                    if cur_dist >= m_d[j] 
                    then (m_inds, m_d, cur_ind, cur_dist)
                    else let cur_dist' = m_d[j]
                         let cur_ind' = m_inds[j]
                         let m_d[j] = cur_dist
                         let m_inds[j] = cur_ind
                         in (m_inds, m_d, cur_ind', cur_dist')
            in (inds', il2s')

let sortPartSortedSeqs [k] (knn: [k](i64,i64)) : [k](i64,i64) =
  -- now knn contains the neighbors in two partially ordered sequences:
  -- one starting at beginning and one starting at the end
  -- we need to sort them
  -- let knn = intrinsics.opaque (copy knn0)
    let (res, _, _) =
        loop (knn_sort, beg, end) = (replicate k (-1i64, i64.highest), 0, k-1)
        for i < k do
            let (next_el, beg', end') =
                if knn[beg].1 < knn[end].1
                then (knn[beg], beg+1, end)
                else (knn[end], beg, end-1)
            let knn_sort[i] = next_el
            in  (knn_sort, beg', end')
    in  res

let bruteForcePar [k] [n]
            (matches: [k]i64)
            (match_dist: [k]i64)
            (candidates: [n]i64)
            (candidate_dist: [n]i64)
            : ([k]i64, [k]i64) =
    let knn = copy (zip matches match_dist)
    let dists = copy candidate_dist
    let cycle = true
    let j = 0i64
    let (_, knn, _, _) =
        loop (dists, knn, j, cycle)
            while cycle && (j < k) do
                let (min_ind, min_val) =
                    reduce_comm (\ (i1,v1) (i2,v2) -> 
                                if v1 < v2 then (i1, v1) else
                                if v1 > v2 then (i2, v2) else
                                (if i1 <= i2 then i1 else i2, v1)
                                ) (n, i64.highest) (zip (iota n) dists)
        
            in  if min_val < (knn[k-1-j].1)
                then  let dists[min_ind] = i64.highest
                    let knn[k-1-j] = (candidates[min_ind], min_val)
                    in  (dists, knn, j+1, true)
                else  (dists, knn, j, false)
    let knn_sort = sortPartSortedSeqs knn
    in  unzip knn_sort