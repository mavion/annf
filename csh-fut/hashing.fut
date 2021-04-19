import "lib/github.com/diku-dk/sobol/sobol-dir-50"
import "lib/github.com/diku-dk/sobol/sobol"
import "lib/github.com/diku-dk/sorts/radix_sort"
import "lib/github.com/diku-dk/cpprandom/random"
import "helper_functions"

module s = Sobol sobol_dir { let D = 2i64 }
module rng_engine = minstd_rand
module rand_f64 = uniform_real_distribution f64 rng_engine

-- taken from https://futhark-lang.org/examples/binary-search.html
let binary_search [n] 't (lte: t -> t -> bool) (xs: [n]t) (x: t) : i64 =
  let (l, _) =
    loop (l, r) = (0, n-1) while l < r do
    let t = l + (r - l) / 2
    in if x `lte` xs[t]
       then (l, t)
       else (t+1, r)
  in l

-- Finds the first <count> sobol numbers and extracts the corresponding representatives for all kernels
let get_repr_sobol 't [k] [n] [m]
                (img: [k][n][m]t)
                (count: i64)
                : [k][count]t =
    let reps = s.sobol count
    in map (\i -> map (\rep ->
        let indx = i64.f64 (rep[0]*(f64.i64 n))
        let indy = i64.f64 (rep[1]*(f64.i64 m))
        in img[i,indx,indy]) reps) (iota k)

-- input kernel projections for src and target image as well as desired number of hash tables
-- only the kernels that are going to be used should be input
-- output is the hashcodes, which are used for constructing and indexing into the hash tables
-- potentially faster version, but not yet optimized or flattened
let create_hash_codes_quants [k] [n] [m] [o] [p]
                (img_a: [k][n][m]i64)
                (img_b: [k][o][p]i64)
                (l: i64)
                (bit_counts: [k]i64)                    
                : ([l][n][m]i64, [l][o][p]i64) =
    -- hardcoded values for 8x8 patch. They are part of the input, but the values for 8x8 patches has been replicated here for convenience
    -- let bit_counts = [5, 3, 3, 1, 1, 1, 2, 2]
    -- let kernels_used = [0, 1, 5, 6, 2, 9, 15, 19]
    -- corresponds to (Y0,0) (Y1,0) (Y0,1) (Y1,1) (Y2,0) (Y0,2) (Cb0,0) (Cr0,0)

    -- compute distributions for each set of kernels k
    -- ground truth is (a ++ b) sorted
    -- approximation is a set of sorted representatives from a and b
    let count_a = 4096 -- TODO figure out good values
    let reps_a = get_repr_sobol img_a count_a
    let reps_b = get_repr_sobol img_b count_a 
    let count_sum = count_a + count_a
    let sort_reps = map2(\x y -> 
                    radix_sort_int i64.num_bits i64.get_bit (concat_to count_sum x y) -- TODO reduce num_bits(a patch of 8x8 patch of 8 bit values shouldn't be larger than 14 bits(15 signed))
                    ) reps_a reps_b
    -- Two possibles approaches: find their (approximate) quantile and then use a hash function to figure out where they'd bin based on the random percentage
    -- or create the bin edges for all l hash tables and then binary search. 
    -- Approach one is O((l+log(representatives))*(nm+op))
    -- Approach two is O((nm+op)*log(bin_sizes)*l)
    -- The costs are similar and the number of bin edges is irregular in approach two, so to avoid having to flatten an irregular segmented binary search the following is approach one.
    let quants_a = map2 (\img sorted -> 
                            map (\row ->
                                map (\p -> 
                                binary_search (<=) sorted p
                                ) row
                            ) img
                        ) img_a sort_reps
    let quants_b = map2 (\img sorted -> 
                            map (\row ->
                                map (\p -> 
                                binary_search (<=) sorted p
                                ) row
                            ) img
                        ) img_b sort_reps
    -- l*k random numbers to create the bin edges for each set of hash tables and their constitutent hashes
    let rands = replicate l (replicate k 1.0)
    let bin_sizes = map (1<<) bit_counts
    let hash_offsets = rotate (-1) (scan (+) 0 bit_counts)
    let hash_offsets[0] = 0
    let rands = map (\x -> map (\i -> rands[x,i] *(f64.i64 (count_sum/bin_sizes[i])) ) (iota k) ) (iota l)
    let hash_codes_a = map  (\x -> 
                                map (\y ->
                                    map (\z -> 
                                        let vals = map(\i -> 
                                            ((quants_a[i,y,z] + i64.f64 rands[x,i])
                                            *bin_sizes[i]/count_sum)<<hash_offsets[i]) (iota k)
                                        in reduce (+) 0 vals
                                    ) (iota m)
                                ) (iota n)
                            ) (iota l)
    let hash_codes_b = map  (\x -> 
                                map (\y ->
                                    map (\z -> 
                                        let vals = map(\i -> 
                                            ((quants_b[i,y,z] + i64.f64 rands[x,i])
                                            *bin_sizes[i]/count_sum)<<hash_offsets[i]) (iota k)
                                        in reduce (+) 0 vals
                                    ) (iota p)
                                ) (iota o)
                            ) (iota l)
    in (hash_codes_a, hash_codes_b)

let create_hash_codes [k] [n] [m]
                (img_a: [k][n]i64)
                (img_b: [k][m]i64)
                (l: i64)
                (bit_counts: [k]i64)                    
                : ([l][n]i64, [l][m]i64) =
    -- hardcoded values for 8x8 patch. They are part of the input, but the values for 8x8 patches has been replicated here for convenience
    -- let bit_counts = [5, 3, 3, 1, 1, 1, 2, 2]
    -- let kernels_used = [0, 1, 5, 6, 2, 9, 15, 19]
    -- corresponds to (Y0,0) (Y1,0) (Y0,1) (Y1,1) (Y2,0) (Y0,2) (Cb0,0) (Cr0,0)

    -- compute distributions for each set of kernels k
    -- arg sort to find the exact position of each patch, from which the correct bin can be found
    let count_sum = m + n
    let arg_sort_both = map2 (\x y -> 
                            radix_sort_int_by_key (\(_,i) -> i)
                            i64.num_bits i64.get_bit -- TODO reduce num_bits(a patch of 8x8 patch of 8 bit values shouldn't be larger than 14 bits(15 signed))
                            (zip (iota count_sum) (concat_to count_sum x y)) 
                        ) img_a img_b
    let both_pos = map (\x ->
                        let (y,_) = unzip x
                        in scatter (replicate count_sum 0) y (iota count_sum)
                        ) arg_sort_both
    let pos_a = both_pos[:,0:n]
    let pos_b = both_pos[:,n:]
    -- pos_a and pos_b contains the index the patch would sort into
    
    -- l*k random numbers to create the bin edges for each set of hash tables and their constitutent hashes
    let rng = rng_engine.rng_from_seed [42]
    let rngs_flat = rng_engine.split_rng (l*k) rng
    let (_,rands_flat) = unzip (map (rand_f64.rand (-0.5f64, 0.5f64)) rngs_flat)
    let rands = unflatten l k rands_flat
    let bin_sizes = map (1<<) bit_counts
    let rands = map (\x -> map (\y -> rands[x,y] * (f64.i64 count_sum) / (f64.i64 bin_sizes[y])) (iota k) ) (iota l)  -- offset is -0.5bin to +0.5bin

    let hash_offsets = rotate (-1) (scan (+) (0) bit_counts)
    let hash_offsets[0] = 0
    let hash_codes_a = map  (\x -> 
                                map (\y ->
                                    let vals = map (\i -> 
                                        let hash_i = ((pos_a[i,y] + i64.f64 rands[x,i])*bin_sizes[i]/count_sum)
                                        let hash_i = if hash_i >= bin_sizes[i] then bin_sizes[i]-1 else hash_i
                                        let hash_i = if hash_i < 0 then 0 else hash_i
                                        in hash_i<<hash_offsets[i]) (iota k)
                                    in reduce (+) 0 vals
                                ) (iota n)
                            ) (iota l)
    let hash_codes_b = map  (\x -> 
                                map (\y ->
                                    let vals = map (\i -> 
                                        let hash_i = ((pos_b[i,y] + i64.f64 rands[x,i])*bin_sizes[i]/count_sum)
                                        let hash_i = if hash_i >= bin_sizes[i] then bin_sizes[i]-1 else hash_i
                                        let hash_i = if hash_i < 0 then 0 else hash_i
                                        in hash_i<<hash_offsets[i]) (iota k)
                                    in reduce (+) 0 vals
                                ) (iota m)
                            ) (iota l)
    in (hash_codes_a, hash_codes_b)

--takes an array of hashcodes, the interval they hash into and the desired width per index
let create_hash_table [n] [l]
                    (hashes_a: [l][n]i64)
                    (span: i64)
                    (width: i64)
                    : ([l][span][width]i64) =
    -- group the codes that hash into the same index
    let arg_sort_hash = map (\xs -> 
                            radix_sort_int_by_key (\(_,i) -> i)
                            i64.num_bits i64.get_bit (zip (iota n) xs) -- TODO reduce num_bits(a hashcode shouldn't be larger than span, which is 18 bits for 8x8)
                        ) hashes_a
    -- pick width codes to keep
    -- ideally random, but the simple approach is to pick the <width> first codes
    let flagss = map (\xs -> map2 (\(_,i) (_,j) -> i != j) (rotate (-1) xs) xs) arg_sort_hash
    let offsetss = map (\flags -> segmented_scan (+) 0 flags (replicate n 1)) flagss
    --let offsetss = map (map(\x -> if x < width then x else -1)) offsetss
    let indicess = map2 (\offsets hashes -> 
                    map2 (\offset (_, hash) -> if offset <= width then hash*width+offset-1 else -1
                    ) offsets hashes
                ) offsetss arg_sort_hash
    let spanwidth = span*width
    let hash_table_flat = map2 (\indices vals -> 
                                let (vals', _) = unzip vals
                                in scatter (replicate spanwidth 0) indices vals') indicess arg_sort_hash
    in map (unflatten span width) hash_table_flat
    -- in hindsight this function probably should've been without the [l], which then could be handled with a map/wrapper function
    -- would be simpler and let the compiler decide how to handle it.