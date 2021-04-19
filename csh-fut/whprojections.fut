import "helper-functions"

-- The recurrence kernel_x[i] = kernel_y[i] o kernel_y[i-stride] o kernel_x[i-stride] is solved here
-- o is plus or minus depending on how prior and current kernel is related
-- Allows for efficient computation of all gray code kernels(except for the first)
let gray_code_step [n] [m] (prior: [n] [m]i64)
                            (stride: i64)
                            (sign: i64)
                            : [n] [m]i64 =
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
                    (init_prj: [n][m]i64)
                    (need_transpose: [o]bool)
                    (strides: [o]i64)
                    (signs: [o]i64)
                    : [o][n][m]i64 = 
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
-- computes the sum for all patches on an image that has been padded with 7 rows/columns of zeros to the top/left
let patch_sum [n] [m]
            (img: [n][m]i64)
            : [n][m]i64 =
    let patch_size = 8
    -- tabulate_2d instead ?
    in map (\x -> 
        map (\y ->
            let x' = if x < patch_size then 0 else x-7
            let y' = if y < patch_size then 0 else y-7
            in reduce (+) 0 (map(reduce (+) 0) (img[x':x+1,y':y+1]))      
        ) (iota m)
    ) (iota n)

-- size is actually [23][n-7*m-7], but functions aren't a valid size types. 
let wh_project [n] [m]
             (img: [n][m][3]i64)
             : [23][]i64 =
    -- create the first gray code projection, which is just a 8x8 convolution of ones.
    -- padding with zeros is used to handle initialization of steps
    let prjs = replicate 23 (replicate (n) (replicate (m) 0))
    -- for each channel compute their simple wh projection. Can't be done efficiently
    let prjs[0,:,:] = patch_sum img[:,:,0]
    let prjs[15,:,:] = patch_sum img[:,:,1]
    let prjs[19,:,:] = patch_sum img[:,:,2]
    -- set hardcoded values corresponding to the gray code steps taken
    -- main channel
    -- 0   <1  <2  <3  <4
    -- ^5  <6  <7  <8
    -- ^9  <10 <11
    -- ^12 <13
    -- ^14
    -- gray code path(1d) is +++ -> ++- -> +-- -> +-+ -> --+
    -- the arrow corresponds to the kernel that the numbered projection will be computed from. I.e. kernel 7 will be computed from kernel 6.
    -- up arrow also requires transposition
    -- let prior       = [ 0, 1, 2, 3, 0, 5, 6, 7, 5, 9,10, 9,12,12]
    let need_transpose = [ false, false, false, false, true, false, false, false, true, false, false, true, false, true]
    let strides        = [ 4, 2, 4, 1, 4, 4, 2, 4, 2, 4, 2, 4, 4, 1]
    let signs          = [-1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1]
    let prjs[1:15,:,:] = gray_code_steps (copy prjs[0,:,:]) need_transpose strides signs
    -- second
    -- 15  <16
    -- ^17 <18
    -- let prior          = [15,15,17]
    let need_transpose = [ false, true, false]
    let strides        = [ 4, 4, 4]
    let signs          = [-1,-1,-1]
    let prjs[16:19,:,:] = gray_code_steps (copy prjs[15,:,:]) need_transpose strides signs
    -- third
    -- 19  <20
    -- ^21 <22
    -- let prior          = [19,19,21]
    let prjs[20:23,:,:] = gray_code_steps (copy prjs[19,:,:]) need_transpose strides signs
    in unflatten 23 ((n-7)*(m-7))(flatten_3d prjs[:,7:,7:])

let wh_project_8bit [n] [m]
             (img: [n][m][3]u8)
             : [23][]i64 =
    wh_project (map (map (map (i64.u8))) img)