import "lib/github.com/diku-dk/sobol/sobol-dir-50"
import "lib/github.com/diku-dk/sobol/sobol"
import "lib/github.com/diku-dk/sorts/radix_sort"
import "lib/github.com/diku-dk/cpprandom/random"
import "helper_functions"

module s = Sobol sobol_dir { let D = 2i64 }
module rng_engine = minstd_rand
module rand_f32 = uniform_real_distribution f32 rng_engine

let create_hash_codes [k] [n] [m]
                (img_a: [k][n]i32)
                (img_b: [k][m]i32)
                (l: i64)
                (bit_counts: [k]i32)                    
                : ([l][n]i32, [l][m]i32) =
    -- hardcoded values for 8x8 patch. They are part of the input, but the values for 8x8 patches has been replicated here for convenience
    -- let bit_counts = [5, 3, 3, 1, 1, 1, 2, 2]
    -- let kernels_used = [0, 1, 5, 6, 2, 9, 15, 19]
    -- corresponds to (Y0,0) (Y1,0) (Y0,1) (Y1,1) (Y2,0) (Y0,2) (Cb0,0) (Cr0,0)

    -- compute distributions for each set of kernels k
    -- arg sort to find the exact position of each patch, from which the correct bin can be found
    let count_sum = m + n
    let arg_sort_both = map2 (\x y -> 
                            radix_sort_int_by_key (\(_,i) -> i)
                            i32.num_bits i32.get_bit -- TODO reduce num_bits(a patch of 8x8 patch of 8 bit values shouldn't be larger than 14 bits(15 signed))
                            (zip (iota count_sum) (concat_to count_sum x y)) 
                        ) img_a img_b
    let both_pos = map (\x ->
                        let (y,_) = unzip x
                        in scatter (replicate count_sum 0) y (map (i32.i64) (iota count_sum))
                        ) arg_sort_both
    let count_sum = i32.i64 count_sum
    let pos_a = both_pos[:,0:n]
    let pos_b = both_pos[:,n:]
    -- pos_a and pos_b contains the index the patch would sort into
    
    -- l*k random numbers to create the bin edges for each set of hash tables and their constitutent hashes
    let rng = rng_engine.rng_from_seed [42]
    let rngs_flat = rng_engine.split_rng (l*k) rng
    let (_,rands_flat) = unzip (map (rand_f32.rand (-0.5f32, 0.5f32)) rngs_flat)
    let rands = unflatten l k rands_flat
    let bin_sizes = map (1<<) bit_counts
    let rands = map (\x -> map (\y -> rands[x,y] * (f32.i32 count_sum) / (f32.i32 bin_sizes[y])) (iota k) ) (iota l)  -- offset is -0.5bin to +0.5bin

    let hash_offsets = rotate (-1) (scan (+) (0) bit_counts)
    let hash_offsets[0] = 0
    let hash_codes_a = map  (\x -> 
                                map (\y ->
                                    let vals = map (\i -> 
                                        let hash_i = ((pos_a[i,y] + i32.f32 rands[x,i])*bin_sizes[i]/count_sum)
                                        let hash_i = if hash_i >= bin_sizes[i] then bin_sizes[i]-1 else hash_i
                                        let hash_i = if hash_i < 0 then 0 else hash_i
                                        in hash_i<<hash_offsets[i]) (iota k)
                                    in reduce (+) 0 vals
                                ) (iota n)
                            ) (iota l)
    let hash_codes_b = map  (\x -> 
                                map (\y ->
                                    let vals = map (\i -> 
                                        let hash_i = ((pos_b[i,y] + i32.f32 rands[x,i])*bin_sizes[i]/count_sum)
                                        let hash_i = if hash_i >= bin_sizes[i] then bin_sizes[i]-1 else hash_i
                                        let hash_i = if hash_i < 0 then 0 else hash_i
                                        in hash_i<<hash_offsets[i]) (iota k)
                                    in reduce (+) 0 vals
                                ) (iota m)
                            ) (iota l)
    in (hash_codes_a, hash_codes_b)

--takes an array of hashcodes, the interval they hash into and the desired width per index
let create_hash_table [n] [l]
                    (hashes_a: [l][n]i32)
                    (span: i64)
                    (width: i64)
                    : ([l][span][width]i32) =
    -- group the codes that hash into the same index
    let arg_sort_hash = map (\xs -> 
                            radix_sort_int_by_key (\(_,i) -> i)
                            i32.num_bits i32.get_bit (zip (iota n) xs) -- TODO reduce num_bits(a hashcode shouldn't be larger than span, which is 18 bits for 8x8)
                        ) hashes_a
    -- pick width codes to keep
    -- ideally random, but the simple approach is to pick the <width> first codes
    let flagss = map (\xs -> map2 (\(_,i) (_,j) -> i != j) (rotate (-1) xs) xs) arg_sort_hash
    let offsetss = map (\flags -> segmented_scan (+) 0 flags (replicate n 1)) flagss
    --let offsetss = map (map(\x -> if x < width then x else -1)) offsetss
    let indicess = map2 (\offsets hashes -> 
                    map2 (\offset (_, hash) -> if offset <= width then (i64.i32 hash)*width+offset-1 else -1
                    ) offsets hashes
                ) offsetss arg_sort_hash
    let spanwidth = span*width
    let hash_table_flat = map2 (\indices vals -> 
                                let (vals', _) = unzip vals
                                in scatter (replicate spanwidth 0) indices (map (i32.i64) vals')) indicess arg_sort_hash
    in map (unflatten span width) hash_table_flat
    -- in hindsight this function probably should've been without the [l], which then could be handled with a map/wrapper function
    -- would be simpler and let the compiler decide how to handle it.