-- Propagation is currently done with i64s, which puts an upper limit on max patch size
-- max bit cost for p8 and p16 is (8+1+6)*2+5=35 and (8+1+8)*2+6=40, so there's room for up to 256*256 patches in i64.
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
let p16:p_constant [44] [3] [8] = {
    kernel_count = 53,
    transpositions_Y = [false,false,false,false,false,false,false,false,
                        true,false,false,false,false,false,false,false,
                        true,false,false,false,false,false,false,
                        true,false,false,false,false,false,
                        true,false,false,false,false,
                        true,false,false,false,
                        true,false,false,
                        true,false,
                        true],
    strides_Y = [8,4,8,2,4,8,4,1, 8,8,4,8,2,4,8,4, 4,8,4,8,2,4,8, 8,8,4,8,2,4, 2,8,4,8,2, 4,8,4,8, 8,8,4, 4,8, 1],
    signs_Y = [-1,-1,1,-1,1,-1,-1,-1, -1,-1,-1,1,-1,1,-1,-1, -1,-1,-1,1,-1,1,-1, 1,-1,-1,1,-1,1, -1,-1,-1,1,-1, 1,-1,-1,1, -1,-1,-1, -1,-1, -1],
    transpositions_C = [ false, true, false],
    strides_C = [ 4, 4, 4],
    signs_C = [-1,-1,-1],
    bit_counts = [5, 3, 3, 1, 1, 1, 2, 2],
    kernels_used = [0, 1, 9, 10, 2, 17, 45, 49], 
    patch_size = 16
}

let p8:p_constant [14] [3] [8] = {
    kernel_count = 23,
    transpositions_Y = [false,false,false,false,true,false,false,false,true,false,false,true,false,true],
    strides_Y = [4,2,4,1, 4,4,2,4, 2,4,2, 4,4, 1],
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
    if p_size == 8       then p8
    else if p_size == 16 then p16
    else                      p1