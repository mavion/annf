-- segmented scan from futhark-book
let segmented_scan 't [n] (g:t->t->t) (ne: t) (flags: [n]bool) (vals: [n]t): [n]t =
  let pairs = scan ( \ (v1,f1) (v2,f2) ->
                       let f = f1 || f2
                       let v = if f2 then v2 else g v1 v2
                       in (v,f) ) (ne,false) (zip vals flags)
  let (res,_) = unzip pairs
  in res

-- segmented reduce. 
let segreduce [n]'t (op: t -> t -> t) (ne:t) (flags: [n]bool) (arr: [n]t): []t =
    let scanres = segmented_scan op ne flags arr
    let int_flags = map (i64.bool ) flags
    let flag_scan = scan (+) 0 int_flags
    -- calculate end index of each segment or out of bounds if it's not the end.
    let inds_or_oob = map2 (\ind b -> if b > 0 then ind-1 else -1) flag_scan (rotate 1 int_flags)
    -- calculate number of segments, scatter needs a destination of exact size.
    let n_segments = if n > 0 then flag_scan[n-1] else 0
    in scatter (replicate n_segments ne) inds_or_oob scanres