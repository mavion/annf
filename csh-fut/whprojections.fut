import "helper_functions"
import "csh_patchsize"

-- The recurrence kernel_x[i] = kernel_y[i] o kernel_y[i-stride] o kernel_x[i-stride] is solved here
-- o is plus or minus depending on how prior and current kernel is related
-- Allows for efficient computation of all gray code kernels(except for the first)
let gray_code_step [n] [m] (prior: [n][m]i32)
                            (stride: i64)
                            (sign: i32)
                            : [n] [m]i32 =
    -- It's solved using scan using linear func comp
    let stride' = stride - 1
    -- map all known/static values
    let afuns = map (\j -> 
                    map (\i -> 
                        if stride' < i 
                        then (prior[j,i] + (sign * prior[j,i-stride]), sign)
                        else (prior[j,i], sign)) (iota m)) (iota n)
    -- segment based on stride
    let sgm_szs = map (\i -> 
                       if i < m % stride
                       then i * (m/stride) + i
                       else i * (m/stride) + m % stride) (iota stride)
    let indices = map (\i -> i / stride + sgm_szs[i % stride] ) (iota m)
    --
    let bfuns = map (\arow -> scatter (replicate m (sign,sign)) indices arow) afuns
    -- segmented scan to handle the recurrence
    let flags = scatter (replicate m false) sgm_szs (replicate stride true)
    let cfuns = map (\as -> 
                    segmented_scan (\(a0,a1) (b0,b1) -> (b0 + b1*a0, a1*b1))
                                   (0,1) 
                                   flags
                                   as) bfuns
    -- reverse the partition
    -- extract the information: True answer is a + b*y0, but y0 is 0 due to padding, as extracting left is enough
    in map (\i ->
            map (\j ->
            let (left, _) = cfuns[i, indices[j]]
            in left) (iota m)) (iota n)

let gray_code_steps [n] [m] [o]
                    (init_prj: [n][m]i32)
                    (need_transpose: [o]bool)
                    (strides: [o]i64)
                    (signs: [o]i32)
                    : [o][n][m]i32 = 
    let (_, _, res) = 
        loop (prior, prior_trs, res) = (init_prj, transpose init_prj, replicate o (replicate n (replicate m 0))) for i < o do
            let next_trs = if need_transpose[i]
                                then gray_code_step prior_trs strides[i] signs[i]
                                else prior_trs
            let next      = if need_transpose[i]
                                then transpose next_trs
                                else gray_code_step prior strides[i] signs[i]
            let res[i,:,:] = next
            in (next, next_trs, res)
    in res
-- computes the sum for all patches on an image that has been padded with patchsize-1 rows/columns of zeros to the top/left
let patch_sum [n] [m]
            (img: [n][m]i32)
            (patch_size: i64)
            : [n][m]i32 =
    -- tabulate_2d instead ?
    map (\x -> 
        map (\y ->
            let x' = if x < patch_size then 0 else x-patch_size+1
            let y' = if y < patch_size then 0 else y-patch_size+1
            in reduce (+) 0 (map(reduce (+) 0) (img[x':x+1,y':y+1]))      
        ) (iota m)
    ) (iota n)

-- size is actually [23][(n-patchsize+1)*(m-patchsize+1)], but functions aren't a valid size types. 
let wh_project [n] [m]
             (img: [n][m][3]i32)
             (p_cons: p_constant [] [] [])
             (k_c: i64)
             : [k_c][]i32 =
    -- create the first gray code projection, which is just a 8x8 convolution of ones.
    -- padding with zeros is used to handle initialization of steps

    let prjs = replicate k_c (replicate (n) (replicate (m) 0))
    -- for each channel compute their simple wh projection. Can't be done efficiently
    let main_size = (length p_cons.transpositions_Y) +1
    let sary_size = (length p_cons.transpositions_C) +1
    let prjs[0,:,:] = patch_sum img[:,:,0] p_cons.patch_size
    let prjs[main_size,:,:] = patch_sum img[:,:,1] p_cons.patch_size
    let prjs[main_size+sary_size,:,:] = patch_sum img[:,:,2] p_cons.patch_size
    -- set hardcoded values corresponding to the gray code steps taken
    let need_transpose = p_cons.transpositions_Y
    let strides        = p_cons.strides_Y
    let signs          = p_cons.signs_Y
    let prjs[1:main_size,:,:] = gray_code_steps (copy prjs[0,:,:]) need_transpose strides signs
    -- second
    let need_transpose = p_cons.transpositions_C
    let strides        = p_cons.strides_C
    let signs          = p_cons.signs_C
    let prjs[main_size+1:main_size+sary_size,:,:] = gray_code_steps (copy prjs[main_size,:,:]) need_transpose strides signs
    -- third
    let prjs[main_size+sary_size+1:k_c,:,:] = gray_code_steps (copy prjs[main_size+sary_size,:,:]) need_transpose strides signs
    let p_sm = p_cons.patch_size - 1
    in unflatten k_c ((n-p_sm)*(m-p_sm))(flatten_3d prjs[:,p_sm:,p_sm:])

let wh_project_8bit [n] [m]
             (img: [n][m][3]u8)
             (p_c: p_constant [] [] [])
             (k_c: i64) 
             : [k_c][]i32 =
    wh_project (map (map (map (i32.u8))) img) p_c k_c