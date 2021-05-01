    -- let kernel_count : i64 =
    --     if      x == 2  then 6
    --     else if x == 4  then 16
    --     if x == 8  then 23
    --     else if x == 16 then 53
    --     else 3 
type p_constant [y] [c] [h] = { kernel_count: i64,
                     transpositions_Y: [y]bool,
                     strides_Y: [y]i64,
                     signs_Y: [y]i64,
                     transpositions_C: [c]bool,
                     strides_C: [c]i64,
                     signs_C: [c]i64,
                     bit_counts: [h]i64,
                     kernels_used: [h]i64,
                     patch_size: i64
}

let p8:p_constant [14] [3] [8] = {
    kernel_count = 23,
    transpositions_Y = [false,false,false,false,true,false,false,false,true,false,false,true,false,true],
    strides_Y = [4,2,4,1,4,4,2,4,2,4,2,4,4,1],
    signs_Y = [-1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1],
    transpositions_C = [ false, true, false],
    strides_C = [ 4, 4, 4],
    signs_C = [-1,-1,-1],
    bit_counts = [5, 3, 3, 1, 1, 1, 2, 2],
    kernels_used = [0, 1, 5, 6, 2, 9, 15, 19], 
    patch_size = 8
}

let p1:p_constant [0] [0] [3] = { -- default/not supported so numbers are meaningless'ish
    kernel_count = 3,
    transpositions_Y = [],
    strides_Y = [],
    signs_Y = [],
    transpositions_C = [],
    strides_C = [],
    signs_C = [],
    bit_counts = [5, 3, 3],
    kernels_used = [0, 1, 2],
    patch_size = 8
}

let get_constants (p_size: i64): p_constant [] [] [] =
    if p_size == 8 then p8
    else                p1