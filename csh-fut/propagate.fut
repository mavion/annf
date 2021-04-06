import "lib/github.com/diku-dk/cpprandom/random"

module rng_engine = minstd_rand
module rand_i64 = uniform_int_distribution i64 rng_engine

let init_matches (fields: i64)
                 (range: i64)
                 : [fields]i64 =
    let rng = rng_engine.rng_from_seed [42]
    let rngs = rng_engine.split_rng fields rng
    let (_,rands) = unzip (map (rand_i64.rand (0, range-1)) rngs)
    in rands



let neighbour_cand [n] [m] [o] [k]
                    (matches: [n]i64)
                    (hash_trg: [m]i64)
                    (hash_table_trg: [o][k]i64)
                    (offset: i64)
                    : [n][]i64 =
    let kplusone = k+1
    in map (\i -> 
            let step = if (((i+offset) < 0) || ((i+offset) >= n)) then 0
                        else (i+offset)
            let matchn = matches[step]
            let match_step = if matchn-offset < 0 || matchn-offset >= m then 0
                                else matchn-offset
            in concat_to kplusone hash_table_trg[hash_trg[match_step],:] [match_step]
        ) (iota n)
    -- neighbours might be out of bounds. In the sequential approach these are ignored and don't add candidates to the propagation
    -- here out of bounds are instead redirected in bounds and add (probably very poor) candidates to keep everything flat.
    -- a different 'flat' approach could be using masks instead.

let find_candidates [n] [m] [o] [k] 
                    (matches: [n]i64)
                    (hash_src: [n]i64)
                    (hash_trg: [m]i64)
                    (hash_table_src: [o][k]i64)
                    (hash_table_trg: [o][k]i64)
                    (dims: (i64,i64))
                    : [n][]i64 =
    -- type 1. check hash_src on table_trg
    let cands = unflatten n (6*k+4) (replicate (n*(6*k+4)) 0)
    let cands[:,0:k] = map (\i -> hash_table_trg[hash_src[i],:]) (iota n)
    let amt = k
    -- type 2. check neighbours->match->neighbour->hash_trg on table_trg
    let (_,y) = dims
    let cands[:,amt:amt+k+1] = neighbour_cand matches hash_trg hash_table_trg 1
    let amt = amt+k+1
    let cands[:,amt:amt+k+1] = neighbour_cand matches hash_trg hash_table_trg (-1)
    let amt = amt+k+1
    let cands[:,amt:amt+k+1] = neighbour_cand matches hash_trg hash_table_trg y
    let amt = amt+k+1
    let cands[:,amt:amt+k+1] = neighbour_cand matches hash_trg hash_table_trg (-y)
    let amt = amt+k+1
    -- type 3. check hash_src on table_src->matches
    let cands[:,amt:amt+k] = map (\i -> map (\j -> matches[j]) hash_table_src[hash_src[i],:]) (iota n)
    in cands

let dist2 [k] (a: [k]i64) (b: [k]i64) : i64 =
    reduce (+) 0 (map2 (\x y -> (x-y)**2) a b)

let best_dist [n] [k] [m]
             (prj_src: [k][n]i64)
             (prj_trg: [k][m]i64)
             (cands: [n][]i64) 
             : [n](i64,i64) =
    map (\i -> 
        let l2s = map (\j -> dist2 prj_src[:,i] prj_trg[:,j]) cands[i,:]
        in reduce (\(x0, x1) (y0, y1) -> 
                    if x0 < y0 then (x0, x1) else (y0, y1)) (i64.highest,0) (zip l2s cands[i,:])
        ) (iota n)

let cmp_cand_match [n]
                    (matc: [n](i64,i64)) -- match is reserved
                    (cand: [n](i64,i64))
                    :[n](i64,i64) =
    map2 (\(x0, x1) (y0, y1) -> if x0 < x1 then (x0, x1) else (y0, y1)) matc cand