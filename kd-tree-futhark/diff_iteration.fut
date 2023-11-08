import "util"
import "diff_bruteforce"

def gather_no_fvs 't [n] (xs: []t) (is: [n]i32) : *[n]t =
  map2 (\i xs' -> xs'[i]) is (replicate n xs)

def dgather_f32 [m][n] (xsbar: [n]f32) (is: [m]i32) (resbar: [m]f32) : *[n]f32 =
  -- map2 (\i xs' -> xs'[i]) is (replicate n xs)
  reduce_by_index (copy xsbar) (+) 0f32 (map i64.i32 is) resbar

def gather_no_fvs_safe 't [m][n] (xs: [m]t) (is: [n]i32) : *[n]t =
  map2 (\i xs' -> if i < i32.i64 m then xs'[i] else xs[0]) is (replicate n xs)

-- TODO add reduce over xs
-- Computation from iterationSorted.
def f [r][n][d][num_leaves][ppl]
      (radiuses: [r]f32)
      (xs: [n][d]f32) -- query
      (x_ws:[n]f32)
      (ys:  [num_leaves][ppl][d]f32)
      (y_ws:  [num_leaves][ppl]f32)
      (leaf_inds: [n]i32)
      (query_inds: [n]i32) =
  let xs = gather xs query_inds
  let x_ws= gather x_ws query_inds
  let new_res0 =
    map3 (\ query query_w leaf_ind ->
            if leaf_ind >= i32.i64 num_leaves
            then replicate r 0.0f32
            else bruteForce radiuses query query_w ys[leaf_ind] y_ws[leaf_ind]
         ) xs x_ws leaf_inds
  -- Semantically equivalent rewrite that differentiates without accumulators:
  -- let xs_sorted   = gather_no_fvs xs query_inds -- NOTE no fvs.
  -- let x_ws_sorted = gather_no_fvs x_ws query_inds
  -- let ys_sorted   = gather_no_fvs_safe ys leaf_inds
  -- let y_ws_sorted = gather_no_fvs_safe y_ws leaf_inds
  -- let new_res0 =
  --   map5 (\ query query_w y y_w leaf_ind ->
  --           if leaf_ind >= i32.i64 num_leaves
  --           then replicate r 0.0f32
  --           else
  --             bruteForce radiuses query query_w y y_w
  --        ) xs_sorted x_ws_sorted ys_sorted y_ws_sorted leaf_inds
  let new_res1 = transpose new_res0
  let new_res = map (reduce (+) 0) new_res1
  in new_res

def df [n][d][num_leaves][ppl][r]
       (radiuses: [r]f32)
       (xs: [n][d]f32) -- query
       (x_ws:[n]f32)
       (ys:  [num_leaves][ppl][d]f32)
       (y_ws:  [num_leaves][ppl]f32)
       (leaf_inds: [n]i32)
       (query_inds: [n]i32)
       -- adjoints:
       -- (xsbar: [n][d]f32)
       (x_ws_bar: [n]f32)
       -- (ysbar: [num_leaves][ppl]f32)
       -- (y_ws_bar: [num_leaves][ppl]f32)
       (resbar: [r]f32) =
  let xs_sorted   = gather_no_fvs xs query_inds -- NOTE no fvs.
  let x_ws_sorted = gather_no_fvs x_ws query_inds
  let ys_sorted   = gather_no_fvs_safe ys leaf_inds
  let y_ws_sorted = gather_no_fvs_safe y_ws leaf_inds
  let new_res0 =
    map5 (\ query query_w y y_w leaf_ind ->
            if leaf_ind >= i32.i64 num_leaves
            then replicate r 0.0f32
            else
              bruteForce radiuses query query_w y y_w
         ) xs_sorted x_ws_sorted ys_sorted y_ws_sorted leaf_inds

  let new_res1 = transpose new_res0
  -- let new_res = map (reduce (+) 0) new_res1 -- Last step unneeded.

  let new_res_bar = resbar
  -- let new_res1_bar =
  --   map3 (\r1 r1bar rbar ->
  --     let _r = reduce (+) 0 r1
  --     let r1bar' = replicate n rbar |> map2 (+) r1bar
  --     in r1bar'
  --   ) new_res1 (replicate r (replicate n 0f32)) new_res_bar
  -- The above is equivalent to the following line.
  let new_res1_bar = map (replicate n) new_res_bar
  let new_res0_bar = transpose new_res1_bar
  let x_ws_sorted_bar = map3 (\(x, x_w, y, y_w) leaf_ind resbar0 ->
    -- Primal unneeded.
    -- Rev.
    let x_w_bar = 0
    let y_w_bar = replicate ppl 0
    let x_w_bar =
      if leaf_ind >= i32.i64 num_leaves
      then x_w_bar + 0f32
      else
        (dbruteForce_opt_seq radiuses x x_w y y_w x_w_bar y_w_bar resbar0).0
    in x_w_bar
  ) (zip4 xs_sorted x_ws_sorted ys_sorted y_ws_sorted) leaf_inds new_res0_bar

  let x_ws_bar = dgather_f32 x_ws_bar query_inds x_ws_sorted_bar
  -- Or using a loop:
  -- let x_ws_bar =
  --   loop x_ws_bar = replicate n 0f32 for k < n do
  --     x_ws_bar with [query_inds[k]] = x_ws_bar[query_inds[k]] + x_ws_sorted_bar[k]
  in x_ws_bar

def df_ALL [n][d][num_leaves][ppl][r]
       (radiuses: [r]f32)
       (xs: [n][d]f32) -- query
       (x_ws:[n]f32)
       (ys:  [num_leaves][ppl][d]f32)
       (y_ws:  [num_leaves][ppl]f32)
       (leaf_inds: [n]i32)
       (query_inds: [n]i32)
       -- adjoints:
       -- (xsbar: [n][d]f32)
       (x_ws_bars: [r][n]f32)
       -- (ysbar: [num_leaves][ppl]f32)
       -- (y_ws_bar: [num_leaves][ppl]f32)
       (resbarsT: [r][r]f32) -- NOTE transposed, but identity matrix transposed is itself!
       : [r][n]f32 =
  let xs_sorted   = gather_no_fvs xs query_inds -- NOTE no fvs.
  let x_ws_sorted = gather_no_fvs x_ws query_inds
  let ys_sorted   = gather_no_fvs_safe ys leaf_inds
  let y_ws_sorted = gather_no_fvs_safe y_ws leaf_inds
  let new_res0 =
    map5 (\ query query_w y y_w leaf_ind ->
            if leaf_ind >= i32.i64 num_leaves
            then replicate r 0.0f32
            else
              bruteForce radiuses query query_w y y_w
         ) xs_sorted x_ws_sorted ys_sorted y_ws_sorted leaf_inds

  let new_res1 = transpose new_res0
  -- let new_res = map (reduce (+) 0) new_res1 -- Last step unneeded.

  let new_res_bars = resbarsT
  -- let new_res1_bar =
  --   map3 (\r1 r1bar rbar ->
  --     let _r = reduce (+) 0 r1
  --     let r1bar' = replicate n rbar |> map2 (+) r1bar
  --     in r1bar'
  --   ) new_res1 (replicate r (replicate n 0f32)) new_res_bar
  -- The above is equivalent to the following line.
  let new_res1_bars = map (replicate n) new_res_bars
  let new_res0_bars = transpose new_res1_bars
  let x_ws_sorted_bar = map3 (\(x, x_w, y, y_w) leaf_ind res0barsT ->
    -- Primal unneeded.
    -- Rev.
    let res0bars: [r][r]f32 = transpose res0barsT -- NOTE transpose here (see resbarsT)!
    let x_w_bar = replicate r 0
    let y_w_bar = replicate r (replicate ppl 0)
    let x_w_bar =
      if leaf_ind >= i32.i64 num_leaves
      then replicate r 0f32
      else
        (dbruteForce_opt_seq_ALL radiuses x x_w y y_w x_w_bar y_w_bar res0bars).0
    in x_w_bar
  ) (zip4 xs_sorted x_ws_sorted ys_sorted y_ws_sorted) leaf_inds new_res0_bars

  let x_ws_bars =
    map2 (\x -> dgather_f32 x query_inds) x_ws_bars (transpose x_ws_sorted_bar)
  -- Or using a loop:
  -- let x_ws_bar =
  --   loop x_ws_bar = replicate n 0f32 for k < n do
  --     x_ws_bar with [query_inds[k]] = x_ws_bar[query_inds[k]] + x_ws_sorted_bar[k]
  in x_ws_bars

-- ==
-- entry: main main_ALL
-- compiled input @ data/5radiuses-iterationSorted-refs-512K-queries-1M.in
-- output { empty([0][3]f32) }
def main [q][n][d][num_leaves][ppl][r]
         (_max_radius: f32)
         (radiuses: [r]f32)
         (_h: i32)
         (_median_dims : [q]i32)
         (_median_vals : [q]f32)
         (_clanc_eqdim : [q]i32)
         (leaves:  [num_leaves][ppl][d]f32)
         (ws:      [num_leaves][ppl]f32)
         (queries: [n][d]f32)
         (query_ws:[n]f32)
         -- the loop state:
         (qleaves:     [n]i32)
         (_stacks:      [n]i32)
         (_dists:       [n]f32)
         (query_inds:  [n]i32)
         (_res:  [r]f32) =
  -- Debugging:
  -- let n = 100000 -- 2048 is num_leaves
  -- let queries = queries[:n]
  -- let query_ws = query_ws[:n]
  -- let qleaves = qleaves[:n]
  -- let query_inds = query_inds[:n]
  -- let query_inds = map (\i -> if i < i32.i64 n then i else 0) query_inds

  -- Testing:
  let out_adj = replicate r 1f32
  let g x_ws = f radiuses queries x_ws leaves ws qleaves query_inds
  let expected = vjp g query_ws out_adj
  let got =
    df radiuses queries query_ws leaves ws qleaves query_inds (replicate n 0f32) out_adj
  let diffs = filter (\(_, x, y) -> x != y) (zip3 (indices expected) expected got)
  in map (\(i, x, y) -> [f32.i64 i, x, y]) diffs

entry main_ALL [q][n][d][num_leaves][ppl][r]
         (_max_radius: f32)
         (radiuses: [r]f32)
         (_h: i32)
         (_median_dims : [q]i32)
         (_median_vals : [q]f32)
         (_clanc_eqdim : [q]i32)
         (leaves:  [num_leaves][ppl][d]f32)
         (ws:      [num_leaves][ppl]f32)
         (queries: [n][d]f32)
         (query_ws:[n]f32)
         -- the loop state:
         (qleaves:     [n]i32)
         (_stacks:      [n]i32)
         (_dists:       [n]f32)
         (query_inds:  [n]i32)
         (_res:  [r]f32) =
  let out_adjs = tabulate r (\i -> (replicate r 0f32) with [i] = 1f32)
  let g x_ws = f radiuses queries x_ws leaves ws qleaves query_inds
  let expected = map (vjp g query_ws) out_adjs
  let got =
    df_ALL radiuses queries query_ws leaves ws qleaves query_inds
           (replicate r (replicate n 0f32)) (transpose out_adjs)
  let expected = flatten expected
  let got = flatten got
  let diffs = filter (\(_, x, y) -> x != y) (zip3 (indices expected) expected got)
  in map (\(i, x, y) -> [f32.i64 i, x, y]) diffs
