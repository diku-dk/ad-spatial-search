import "util"
import "diff_bruteforce"

def gather_no_fvs 't [n] (xs: []t) (is: [n]i32) : *[n]t =
  map2 (\i xs' -> xs'[i]) is (replicate n xs)

-- TODO can be optimised if we assume no duplicates in is.
-- (Just a map over xs zeroing any indices not in is then?)
def dgather_f32 [m][n] (xsbar: [n]f32) (is: [m]i32) (resbar: [m]f32) : *[n]f32 =
  -- map2 (\i xs' -> xs'[i]) is (replicate n xs)
  reduce_by_index (copy xsbar) (+) 0f32 (map i64.i32 is) resbar

def dgather2d_f32 [m][n][d] (xsbar: [n][d]f32) (is: [m]i32) (resbar: [m][d]f32) : *[n][d]f32 =
  -- map2 (\i xs' -> xs'[i]) is (replicate n xs)
  reduce_by_index (copy xsbar) (map2 (+)) (replicate d 0f32) (map i64.i32 is) resbar

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
  -- let xs   = gather_no_fvs xs query_inds -- NOTE no fvs.
  -- let x_ws = gather_no_fvs x_ws query_inds
  -- let ys   = gather_no_fvs_safe ys leaf_inds
  -- let y_ws = gather_no_fvs_safe y_ws leaf_inds
  -- let new_res0 =
  --   map5 (\ query query_w y y_w leaf_ind ->
  --           if leaf_ind >= i32.i64 num_leaves
  --           then replicate r 0.0f32
  --           else bruteForce radiuses query query_w y y_w
  --        ) xs x_ws ys y_ws leaf_inds

  let new_res1 = transpose new_res0
  let new_res = map (reduce (+) 0.0f32) new_res1
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
  let xs   = gather_no_fvs xs query_inds -- NOTE no fvs.
  let x_ws = gather_no_fvs x_ws query_inds
  let ys   = gather_no_fvs_safe ys leaf_inds
  let y_ws = gather_no_fvs_safe y_ws leaf_inds

  let new_res0 =
    map5 (\ query query_w y y_w leaf_ind ->
            if leaf_ind >= i32.i64 num_leaves
            then replicate r 0.0f32
            else bruteForce radiuses query query_w y y_w
         ) xs x_ws ys y_ws leaf_inds
  let new_res1 = transpose new_res0
  -- let new_res = map (reduce (+) 0.0f32) new_res1 -- Last step unneeded.

  -- NOTE For now: differentiate w.r.t. query_ws only.
  -- which is not a free variable, so I will attempt to just ignore free variables
  -- here.
  -- TODO diff gathers

  -- Simplifying zeros:
  let new_res_bar = resbar

  let new_res1_bar = replicate r (replicate n 0f32) -- TODO function arg bc outer loop?
  let new_res1_bar = map3 (\_as asbar rbar ->
    replicate n rbar |> map2 (+) asbar
  ) new_res1 new_res1_bar new_res_bar

  let new_res0_bar = transpose new_res1_bar

  let x_ws_bar = map5 (\x x_w leaf_ind x_w_bar resbar0 ->
    -- Primal unneeded.
    -- Rev.
    let resbar = resbar0
    let x_w_bar =
      if leaf_ind >= i32.i64 num_leaves
      then x_w_bar + 0f32
      else
        (dbruteForce_opt_seq radiuses x x_w ys[leaf_ind] y_ws[leaf_ind] x_w_bar y_ws_bar[leaf_ind] resbar).0
    in x_w_bar
  ) xs x_ws leaf_inds x_ws_bar new_res0_bar
  let _new_res0_bar = x_ws_bar -- Only output from above; should be tuple of above, I think?

  let x_ws_bar = #[trace] x_ws_bar
  let x_ws_bar = dgather_f32 x_ws_bar query_inds x_ws_bar
  -- TODO how to deal with this; y_ws_bar outerdim is not same as leaf_inds?
  -- let y_ws_bar = dgather_f32 y_ws_bar leaf_inds y_ws_bar

  -- Unneeded:
  -- let xsbar = dgather2d_f32 xsbar query_inds xsbar -- Doesn't depend on this.
  -- let ysbar = dgather2d_f32 ysbar leaf_inds ysbar -- Doesn't depend on this.

  in x_ws_bar

-- ==
-- compiled input @ data/5radiuses-iterationSorted-refs-512K-queries-1M.in
-- output { true }
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
  let n = 10000
  let queries = queries[:n]
  let query_ws = query_ws[:n]
  let qleaves = qleaves[:n]
  let query_inds = query_inds[:n]
  let query_inds = map (\i -> if i < i32.i64 n then i else 0) query_inds

  -- Testing:
  let out_adj = replicate r 1f32
  let x_ws_bar = replicate n 0f32
  let y_ws_bar = replicate num_leaves (replicate ppl 0f32)
  let g x_ws = f radiuses queries x_ws leaves ws qleaves query_inds
  let expected = vjp g query_ws out_adj
  let got =
    df radiuses queries query_ws leaves ws qleaves query_inds
                     x_ws_bar y_ws_bar out_adj
  let expected = #[trace(expected)] expected[:10]
  let got = #[trace(got)] got[:10]
  in map2 (==) expected got |> reduce (&&) true
