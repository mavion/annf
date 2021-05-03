import "propagate"
import "helper_functions"

-- Benchmarking computing distances for all candidates. 
-- Used to check for coalesced access
-- ==
-- entry: dist2_prop dist2_part_flat
-- compiled input @ l2_data.in

entry dist2_prop [patch_count] [m] [k] [l]
    (wh_src_trs: [patch_count][k]i64)
    (wh_trg_trs: [m][k]i64)
    (candidates: [patch_count][l]i64) =
    map2 (\x ys -> map (\y -> dist2 wh_src_trs[x,:] wh_trg_trs[y,:]) ys) (iota patch_count) candidates

entry dist2_part_flat [patch_count] [m] [k] [l]
    (wh_src_trs: [patch_count][k]i64)
    (wh_trg_trs: [m][k]i64)
    (candidates: [patch_count][l]i64) =
    dist2_all wh_src_trs wh_trg_trs candidates
    -- let src_points = map (/l) (iota (patch_count*l))
    -- let cands_f = flatten_to (patch_count*l) candidates
    -- let dists = map2 (\x y -> dist2 wh_src_trs[x,:] wh_trg_trs[y,:]) src_points cands_f
    -- in unflatten patch_count l dists

-- out of memory error
-- entry dist2_flat [patch_count] [m] [k] [l]
--     (wh_src_trs: [patch_count][k]i64)
--     (wh_trg_trs: [m][k]i64)
--     (candidates: [patch_count][l]i64) =
--     let cand_count = patch_count*l
--     let src_points = map (/l) (iota cand_count)
--     let cands_f = flatten_to cand_count candidates
--     let point_count = cand_count*k
--     let src_vals = flatten_to point_count (map (\x -> wh_src_trs[x,:]) src_points)
--     let trg_vals = flatten_to point_count (map (\x -> wh_trg_trs[x,:]) cands_f)
--     let vals = map2 (\x y -> (x-y)*(x-y)) src_vals trg_vals
--     let flags = map (\x -> x%k == 0) (iota point_count)
--     let dists = segreduce (+) 0 flags vals
--     in unflatten patch_count l dists