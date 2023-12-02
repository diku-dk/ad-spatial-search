import "util"
import "diff_bruteforce"

def gather_no_fvs 't [n] (xs: []t) (is: [n]i32) : *[n]t =
  map2 (\i xs' -> xs'[i]) is (replicate n xs)

def dgather_f32 [m][n] (xsbar: [n]f32) (is: [m]i32) (resbar: [m]f32) : *[n]f32 =
  -- map2 (\i xs' -> xs'[i]) is (replicate n xs)
  reduce_by_index (copy xsbar) (+) 0f32 (map i64.i32 is) resbar

def gather_no_fvs_safe 't [m][n] (xs: [m]t) (is: [n]i32) : *[n]t =
  map2 (\i xs' -> if i < i32.i64 m then xs'[i] else xs[0]) is (replicate n xs)

-- TODO could make output of dbruteForce_opt_seq_ALL already transposed then
-- we save transposes here and also in dbruteForce_opt_seq_ALL itself...
def dgather_safe_f32 [m][n][p] (xssbar: [m][p]f32) (is: [n]i32) (yssbar: [n][p]f32): *[m][p]f32 =
  -- reduce_by_index (copy xssbar) (map2 (+)) (replicate p 0f32) (map i64.i32 is) yssbar
  -- ihwim rewrite enjoys a big speed-up:
  map2 (\xsbarT resbarT ->
    reduce_by_index (copy xsbarT) (+) 0f32 (map i64.i32 is) resbarT
  ) (transpose xssbar) (transpose yssbar)
  |> transpose

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
  -- let xs_sorted   = gather_no_fvs xs query_inds -- no fvs.
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
       (y_ws_bar: [num_leaves][ppl]f32)
       (resbar: [r]f32) =
  -- Primal.
  -- Semantically equivalent rewrite that differentiates without accumulators:
  let xs_sorted   = gather_no_fvs xs query_inds -- no fvs.
  let x_ws_sorted = gather_no_fvs x_ws query_inds
  let ys_sorted   = gather_no_fvs_safe ys leaf_inds
  let y_ws_sorted = gather_no_fvs_safe y_ws leaf_inds
  -- The rest of the primal is unneeded.
  -- let new_res0 =
  --   map5 (\ query query_w y y_w leaf_ind ->
  --           if leaf_ind >= i32.i64 num_leaves
  --           then replicate r 0.0f32
  --           else
  --             bruteForce radiuses query query_w y y_w
  --        ) xs_sorted x_ws_sorted ys_sorted y_ws_sorted leaf_inds
  -- let new_res1 = transpose new_res0
  -- let new_res = map (reduce (+) 0) new_res1

  -- Rev.
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
  let (x_ws_sorted_bar, y_ws_sorted_bar) =
    map3 (\(x, x_w, y, y_w) leaf_ind resbar0 ->
      -- Primal unneeded.
      -- Rev.
      let x_w_bar = 0
      let y_w_bar = replicate ppl 0
      let (x_w_bar, y_w_bar) =
        if leaf_ind >= i32.i64 num_leaves
        then (x_w_bar, y_w_bar)
        else dbruteForce_opt_seq radiuses x x_w y y_w x_w_bar y_w_bar resbar0
      in (x_w_bar, y_w_bar)
    ) (zip4 xs_sorted x_ws_sorted ys_sorted y_ws_sorted) leaf_inds new_res0_bar
    |> unzip

  let x_ws_bar = dgather_f32 x_ws_bar query_inds x_ws_sorted_bar
  -- Or using a loop:
  -- let x_ws_bar =
  --   loop x_ws_bar = replicate n 0f32 for k < n do
  --     x_ws_bar with [query_inds[k]] = x_ws_bar[query_inds[k]] + x_ws_sorted_bar[k]
  let y_ws_bar = dgather_safe_f32 y_ws_bar leaf_inds y_ws_sorted_bar
  in (x_ws_bar, y_ws_bar)

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
       (y_ws_bars: [r][num_leaves][ppl]f32)
       (resbarsT: [r][r]f32) -- NOTE transposed, but identity matrix transposed is itself!
       : ([r][n]f32, [r][num_leaves][ppl]f32) =
  -- Primal.
  let xs_sorted   = gather_no_fvs xs query_inds -- NOTE no fvs.
  let x_ws_sorted = gather_no_fvs x_ws query_inds
  let ys_sorted   = gather_no_fvs_safe ys leaf_inds
  let y_ws_sorted: [n][ppl]f32 = gather_no_fvs_safe y_ws leaf_inds
  -- The rest of the primal is unneeded.
  -- let new_res0 =
  --   map5 (\ query query_w y y_w leaf_ind ->
  --           if leaf_ind >= i32.i64 num_leaves
  --           then replicate r 0.0f32
  --           else
  --             bruteForce radiuses query query_w y y_w
  --        ) xs_sorted x_ws_sorted ys_sorted y_ws_sorted leaf_inds

  -- let _new_res1 = transpose new_res0
  -- let new_res = map (reduce (+) 0) new_res1 -- Last step unneeded.

  -- Rev.
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
  -- y_ws_sorted_bar are computed transposed below for efficiency.
  let (x_ws_sorted_bar: [n][r]f32, y_ws_sorted_barT: [n][ppl][r]f32) =
    map3 (\(x, x_w, y, y_w) leaf_ind res0barsT ->
      -- Primal unneeded.
      -- Rev.
      let res0bars: [r][r]f32 = transpose res0barsT -- NOTE transpose here (see resbarsT)!
      let x_w_bar = replicate r 0
      let y_w_barT = replicate ppl (replicate r 0)
      let (x_w_bar: [r]f32, y_w_barT: [ppl][r]f32) =
        if leaf_ind >= i32.i64 num_leaves
        then (x_w_bar, y_w_barT)
        else
          #[sequential] dbruteForce_opt_seq_ALL_T radiuses x x_w y y_w x_w_bar y_w_barT res0bars
      in (x_w_bar, y_w_barT)
    ) (zip4 xs_sorted x_ws_sorted ys_sorted y_ws_sorted) leaf_inds new_res0_bars
    |> unzip

  -- Map over radius dimension.
  let x_ws_bars =
    map2 (\x -> dgather_f32 x query_inds) x_ws_bars (transpose x_ws_sorted_bar)
  let y_ws_bars =
    map2 (\x -> dgather_safe_f32 x leaf_inds) y_ws_bars (transpose (map transpose y_ws_sorted_barT))
  in (x_ws_bars, y_ws_bars)

-- ==
-- entry: main
-- nobench compiled input @ data/5radiuses-iterationSorted-refs-512K-queries-1M.in
-- output { empty([0][3]f32) empty([0][4]f32) }
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
  -- Debugging query_ws:
  -- let n = 100000 -- 2048 is num_leaves
  -- let queries = queries[:n]
  -- let query_ws = query_ws[:n]
  -- let qleaves = qleaves[:n]
  -- let query_inds = query_inds[:n]
  -- let query_inds = map (\i -> if i < i32.i64 n then i else 0) query_inds

  -- Debugging ws:
  -- let num_leaves = 1
  -- let leaves = leaves[:num_leaves]
  -- let ws = ws[:num_leaves]

  -- Testing:
  let out_adj = replicate r 1f32
  let g (x_ws, y_ws) = f radiuses queries x_ws leaves y_ws qleaves query_inds
  let (expected_x, expected_y) = vjp g (query_ws, ws) out_adj
  let (got_x, got_y) =
    df radiuses queries query_ws leaves ws qleaves query_inds (replicate n 0f32) (replicate num_leaves (replicate ppl 0f32)) out_adj
  let diffs_x = filter (\(_, x, y) -> x != y) (zip3 (indices expected_x) expected_x got_x)
  let expected_y = flatten expected_y
  let got_y = flatten got_y
  let inds = map (\i -> map (\j -> (i,j)) (iota ppl)) (iota num_leaves)
             |> flatten
  -- Note can't do exact equality here.
  let diffs_y = filter (\(_, x, y) -> f32.abs (x - y) > 1e-5) (zip3 inds expected_y got_y)
  in (map (\(i, x, y) -> [f32.i64 i, x, y]) diffs_x,
      map (\((i,j), x, y) -> [f32.i64 i, f32.i64 j, x, y]) diffs_y)

-- ==
-- entry: main_ALL
-- nobench compiled input @ data/5radiuses-iterationSorted-refs-512K-queries-1M.in
-- output { empty([0][3]f32) empty([0][3]f32) }
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
  let g (x_ws, y_ws) = f radiuses queries x_ws leaves y_ws qleaves query_inds
  let (expected_x, expected_y) = unzip <| map (vjp g (query_ws, ws)) out_adjs
  let (got_x, got_y) =
    df_ALL radiuses queries query_ws leaves ws qleaves query_inds
           (replicate r (replicate n 0f32)) (replicate r (replicate num_leaves (replicate ppl 0f32)))
           out_adjs
           --^ out_adjs.T = out_adjs

  let expected_x = flatten expected_x
  let got_x = flatten got_x
  let diffs_x = filter (\(_, x, y) -> x != y) (zip3 (indices expected_x) expected_x got_x)

  let expected_y = flatten_3d expected_y
  let got_y = flatten_3d got_y
  -- Note can't do exact equality here.
  let diffs_y = filter (\(_, x, y) -> f32.abs (x - y) > 1e-5) (zip3 (indices expected_y) expected_y got_y)
  in (map (\(i, x, y) -> [f32.i64 i, x, y]) diffs_x,
      map (\(i, x, y) -> [f32.i64 i, x, y]) diffs_y)

-- ==
-- entry: bench_manual bench_ad
-- compiled input @ data/5radiuses-iterationSorted-refs-512K-queries-1M.in
entry bench_manual [q][n][d][num_leaves][ppl][r]
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
  in df_ALL radiuses queries query_ws leaves ws qleaves query_inds
            (replicate r (replicate n 0f32)) (replicate r (replicate num_leaves (replicate ppl 0f32)))
            out_adjs
            --^ out_adjs.T = out_adjs

entry bench_ad [q][n][d][num_leaves][ppl][r]
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
  in map (vjp g query_ws) out_adjs

-- Timings when only computing x_ws_bar.
--
-- [tvk568@futharkhpa03fl kd-tree-futhark]$ ./futhark-nightly/bin/futhark bench --backend=cuda --pass-option=--device=#1 diff_iteration.fut
-- diff_iteration.fut:bench_manual (no tuning file):
-- data/5radiuses-iterationSorted-refs-5...:      11904μs (95% CI: [   11900.0,    11909.0])
-- diff_iteration.fut:bench_ad (no tuning file):
-- data/5radiuses-iterationSorted-refs-5...:     266417μs (95% CI: [  265753.0,   267107.0])
--
-- Timings when computing x_ws_bar and y_ws_bar.
-- diff_iteration.fut:bench_manual (no tuning file):
-- data/5radiuses-iterationSorted-refs-5...:      67096μs (95% CI: [   66838.3,    67822.7])
-- diff_iteration.fut:bench_ad (no tuning file):
-- data/5radiuses-iterationSorted-refs-5...:     264638μs (95% CI: [  264043.2,   265328.8])
