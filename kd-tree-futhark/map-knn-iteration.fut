import "lib/github.com/diku-dk/sorts/radix_sort"

import "kd-traverse"
import "util"
import "diff_bruteforce"

def bruteForce [m][d][r]
               (radiuses: [r]f32)
               (x: [d]f32) -- One point from sample 1.
               (x_w: f32)
               (ys: [m][d]f32) -- Sample 2.
               (y_ws: [m]f32)
               : [r]f32 =
    -- map2(\y y_w ->
    --       let dist = sumSqrsSeq x y
    --       in  map (\radius -> if dist <= radius then x_w * y_w else 0.0f32) radiuses
    --     ) ys y_ws
    --- |> reduce (map2 (+)) (replicate r 0.0f32)
    loop res = replicate r 0f32 for i < m do
      let (y, y_w) = (ys[i], y_ws[i])
      let dist =
        loop s = 0f32 for j < d do
          let z = x[j] - y[j]
          in s + z*z
      let wprod = x_w * y_w
      in loop res' = res for k in (reverse (iota r)) do
           if dist <= radiuses[k]
           then res' with [k] = res'[k] + wprod
           else res'

def dbruteForce [m][d][r]
                (radiuses: [r]f32)
                (x: [d]f32) -- One point from sample 1.
                (x_w: f32)
                (ys: [m][d]f32) -- Sample 2.
                (y_ws: [m]f32)
                (xbar_w: f32)
                (ybar_ws: [m]f32)
                (res_bar: [r]f32)
                : (f32, [m]f32) =
  dbruteForce_opt_seq radiuses x x_w ys y_ws xbar_w ybar_ws res_bar
  -- dbruteForce_opt_soacs radiuses x x_w ys y_ws res_bar

def sortQueriesByLeavesRadix [n] (num_bits: i32) (leaves: [n]i32) : ([n]i32, [n]i32) =
  -- (leaves, map i32.i64 (iota n))
  unzip <| radix_sort_by_key (\(l,_) -> l) num_bits i32.get_bit (zip leaves (map i32.i64 (iota n)))

def iterationSorted [q][n][d][num_leaves][ppl][r]
            (max_radius: f32)
            (radiuses: [r]f32)
            (h: i32)
            (kd_tree: [q](i32,f32,i32))
            (leaves:  [num_leaves][ppl][d]f32)
            (ws:      [num_leaves][ppl]f32)
            -- ^ invariant
            (queries: [n][d]f32)
            (query_ws:[n]f32)
            -- the loop state:
            (qleaves:     [n]i32)
            (stacks:      [n]i32)
            (dists:       [n]f32)
            (query_inds:  [n]i32)
            (res:  [r]f32)
          : ([n]i32, [n]i32, [n]f32, [n]i32, [r]f32) =

  let queries_sorted = gather queries  query_inds
  let query_ws_sorted= gather query_ws query_inds

  -- apply brute force
  let new_res =
    map3 (\ query query_w leaf_ind ->
            if leaf_ind >= i32.i64 num_leaves
            then replicate r 0.0f32
            else bruteForce radiuses query query_w (leaves[leaf_ind]) (ws[leaf_ind])
         ) queries_sorted query_ws_sorted qleaves
    |> transpose |> map (reduce (+) 0.0f32)

  -- start at old leaf and find a new leaf, until done!
  let (new_leaves, new_stacks, new_dists) = unzip3 <|
    map4 (\ query leaf_ind stack dist ->
            if leaf_ind >= i32.i64 num_leaves
            then
                 (leaf_ind, stack, dist)
            else traverseOnce max_radius h kd_tree query
                              (leaf_ind, stack, dist)
         ) queries_sorted qleaves stacks dists

  let (qleaves', sort_inds) = sortQueriesByLeavesRadix (h+2) new_leaves
  -- we need (h+2) bits because the finish leaf is represented by num_leaves

  --let num_valid = map (\l -> if l < i32.i64 num_leaves then 1 else 0) new_leaves_all
  --                |> reduce_comm (+) 0i32 |> i64.i32

  let stacks'  = gather new_stacks sort_inds
  let dists'   = gather new_dists  sort_inds
  let query_inds' = gather query_inds sort_inds

  in  (qleaves', stacks', dists', query_inds', map2 (+) res new_res)

import "diff_iteration"

def diterationSorted [q][n][d][num_leaves][ppl][r]
      (max_radius: f32)
      (radiuses: [r]f32)
      (h: i32)
      (kd_tree: [q](i32,f32,i32))
      (leaves:  [num_leaves][ppl][d]f32)
      (ws:  [num_leaves][ppl]f32)
      -- ^ invariant
      (queries: [n][d]f32)
      (query_ws:[n]f32)
      -- the loop state:
      (qleaves:     [n]i32)
      (stacks:      [n]i32)
      (dists:       [n]f32)
      (query_inds:  [n]i32)
      (res:  [r]f32)
      -- adjoints:
      (query_ws_bar:  [n]f32)         -- x_ws
      -- (ws_bar:  [num_leaves][ppl]f32) -- y_ws
      (resbar:  [r]f32)
      : ([n]i32, [n]i32, [n]f32, [n]i32, [r]f32, [n]f32, [r]f32) =
  -- Run primal for control-flow variables.
  let (qleaves', stacks', dists', query_inds', new_res) =
    iterationSorted max_radius radiuses h kd_tree leaves ws queries
                    query_ws qleaves stacks dists query_inds res
  let (new_res_bar, query_ws_bar) =
    (resbar, -- NOTE resbar does not change.
     df radiuses queries query_ws leaves ws qleaves query_inds query_ws_bar resbar)

  in (qleaves', stacks', dists', query_inds', new_res, query_ws_bar, new_res_bar)

def diterationSorted_ALL [q][n][d][num_leaves][ppl][r]
      (max_radius: f32)
      (radiuses: [r]f32)
      (h: i32)
      (kd_tree: [q](i32,f32,i32))
      (leaves:  [num_leaves][ppl][d]f32)
      (ws:  [num_leaves][ppl]f32)
      -- ^ invariant
      (queries: [n][d]f32)
      (query_ws:[n]f32)
      -- the loop state:
      (qleaves:     [n]i32)
      (stacks:      [n]i32)
      (dists:       [n]f32)
      (query_inds:  [n]i32)
      (res:  [r]f32)
      -- adjoints:
      (query_ws_bar: [r][n]f32)         -- x_ws
      -- (ws_bar:  [num_leaves][ppl]f32) -- y_ws
      (resbar:  [r][r]f32)
      : ([n]i32, [n]i32, [n]f32, [n]i32, [r]f32, [r][n]f32, [r][r]f32) =
  -- Run primal for control-flow variables.
  let (qleaves', stacks', dists', query_inds', new_res) =
    iterationSorted max_radius radiuses h kd_tree leaves ws queries
                    query_ws qleaves stacks dists query_inds res
  let (new_res_bar, query_ws_bar) =
    (resbar, -- NOTE resbar does not change.
     df_ALL radiuses queries query_ws leaves ws qleaves query_inds query_ws_bar resbar)
  in (qleaves', stacks', dists', query_inds', new_res, query_ws_bar, new_res_bar)

-- Eliminate duplicate work by inlining primal and df.
def diterationSorted_ALL_inlined [q][n][d][num_leaves][ppl][r]
      (max_radius: f32)
      (radiuses: [r]f32)
      (h: i32)
      (kd_tree: [q](i32,f32,i32))
      (leaves:  [num_leaves][ppl][d]f32)
      (ws:  [num_leaves][ppl]f32)
      -- ^ invariant
      (queries: [n][d]f32)
      (query_ws:[n]f32)
      -- the loop state:
      (qleaves:     [n]i32)
      (stacks:      [n]i32)
      (dists:       [n]f32)
      (query_inds:  [n]i32)
      -- adjoints:
      (query_ws_bar: [r][n]f32)         -- x_ws
      -- (ws_bar:  [num_leaves][ppl]f32) -- y_ws
      (resbars:  [r][r]f32)
      : ([n]i32, [n]i32, [n]f32, [n]i32, [r][n]f32, [r][r]f32) =
  -- PRIMAL (new_res modified to keep intermediate results).
  let queries_sorted = gather queries  query_inds
  let query_ws_sorted= gather query_ws query_inds

  -- apply brute force
  -- let new_res0 =
  --   map5 (\ query query_w y y_w leaf_ind ->
  --           if leaf_ind >= i32.i64 num_leaves
  --           then replicate r 0.0f32
  --           else bruteForce radiuses query query_w y y_w
  --        ) queries_sorted query_ws_sorted ys_sorted y_ws_sorted qleaves
  -- let new_res = new_res0 |> transpose |> map (reduce (+) 0.0f32)

  -- start at old leaf and find a new leaf, until done!
  let (new_leaves, new_stacks, new_dists) = unzip3 <|
    map4 (\ query leaf_ind stack dist ->
            if leaf_ind >= i32.i64 num_leaves
            then
                 (leaf_ind, stack, dist)
            else traverseOnce max_radius h kd_tree query
                              (leaf_ind, stack, dist)
         ) queries_sorted qleaves stacks dists
    |> opaque

  let (qleaves', sort_inds) = sortQueriesByLeavesRadix (h+2) new_leaves
  -- we need (h+2) bits because the finish leaf is represented by num_leaves

  let stacks'  = gather new_stacks sort_inds
  let dists'   = gather new_dists  sort_inds
  let query_inds' = gather query_inds sort_inds

  -- let new_res = map2 (+) res new_res

  -- DERIVATIVE.
  -- Renaming:
  let xs_sorted = queries_sorted
  let x_ws_sorted = query_ws_sorted
  let leaf_inds = qleaves
  let x_ws_bars = query_ws_bar

  let ys_sorted = gather_no_fvs_safe leaves qleaves
  let y_ws_sorted = gather_no_fvs_safe ws qleaves

  let new_res_bars = resbars -- Actually transposed here.
  let new_res1_bars = map (replicate n) new_res_bars
  let new_res0_bars = transpose new_res1_bars
  let x_ws_sorted_bar = map3 (\(x, x_w, y, y_w) leaf_ind res0barsT ->
    -- Primal unneeded.
    -- Rev.
    let res0bars: [r][r]f32 = transpose res0barsT -- NOTE transpose.
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

  let query_ws_bar = x_ws_bars
  let new_res_bar = resbars -- NOTE unchanged.

  in (qleaves', stacks', dists', query_inds', query_ws_bar, new_res_bar)
