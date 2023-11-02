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
    |> transpose |> map (reduce (+) 0.0f32) |> opaque

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

  --let num_valid = map (\l -> if l < i32.i64 num_leaves then 1 else 0) new_leaves_all
  --                |> reduce_comm (+) 0i32 |> i64.i32

  let stacks'  = gather new_stacks sort_inds
  let dists'   = gather new_dists  sort_inds
  let query_inds' = gather query_inds sort_inds

  in  (qleaves', stacks', dists', query_inds', map2 (+) res new_res)

def diff_iterationSorted [q][n][d][num_leaves][ppl][r]
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
      (ws_bar:  [num_leaves][ppl]f32) -- y_ws
      (resbar:  [r]f32)
      : ([n]i32, [n]i32, [n]f32, [n]i32, [r]f32, [n]f32) =
  -- Run primal for control-flow variables.
  let (qleaves', stacks', dists', query_inds', res_ws') =
    iterationSorted max_radius radiuses h kd_tree leaves ws queries
                    query_ws qleaves stacks dists query_inds res
  -- TODO copy paste above primal

  -- The part of the primal that actually depends on weights is `new_res`.
  let queries_sorted = gather queries  query_inds
  let query_ws_sorted= gather query_ws query_inds

  -- apply brute force
  let new_res0 =
    map3 (\ query query_w leaf_ind ->
            if leaf_ind >= i32.i64 num_leaves
            then replicate r 0.0f32
            else bruteForce radiuses query query_w (leaves[leaf_ind]) (ws[leaf_ind])
         ) queries_sorted query_ws_sorted qleaves
  let new_res1 = transpose new_res0
  -- let new_res = map (reduce (+) 0.0f32) new_res1 -- Last step unneeded.

  -- TODO Differentiate w.r.t. ws and query_ws.
  -- The map for new_res has free variables (in particular, ws is a free variable).
  -- This requires special handling (replication or withAcc) see pp 7 in paper.
  -- NOTE For now: differentiate w.r.t. query_ws only.
  -- which is not a free variable, so I will attempt to just ignore free variables
  -- here.

  -- TODO diff gathers

  -- let new_res_bar = resbar
  -- let new_res1_bar = replicate n 0f32
  -- let new_res1_bar = map3 (\as asbar rbar ->
  --   let r = reduce (+) 0.0f32 as
  --   let asbar' = replicate n rbar |> map (+) asbar
  --   in asbar + asbar'
  -- ) new_res1 new_res1_bar new_res_bar
  --
  -- Simplifying zeros:
  let new_res_bar = resbar
  let new_res1_bar = map2 (\_as rbar ->
    -- let r = reduce (+) 0.0f32 as
    let asbar' = replicate n rbar -- TODO is n the right size here?
    in asbar'
  ) new_res1 new_res_bar

  let new_res0_bar = transpose new_res1_bar
  let new_res0_bar = map5 (\query query_w leaf_ind query_w_bar resbar0 ->
    -- Primal unneeded.
    -- let res =
    --   if leaf_ind >= i32.i64 num_leaves
    --   then replicate r 0.0f32
    --   else bruteForce radiuses query query_w (leaves[leaf_ind]) (ws[leaf_ind])
    -- Rev.
    let resbar = resbar0
    let query_w_bar =
      if leaf_ind >= i32.i64 num_leaves
      then query_w_bar + 0f32
      else
        (dbruteForce radiuses query query_w leaves[leaf_ind] ws[leaf_ind] query_w_bar ws_bar[leaf_ind] resbar).0
    in query_w_bar
  ) queries_sorted query_ws_sorted qleaves query_ws_bar new_res0_bar

  in (qleaves', stacks', dists', query_inds', res_ws', new_res0_bar)
