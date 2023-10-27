import "lib/github.com/diku-dk/sorts/radix_sort"

import "kd-traverse"
import "util"

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
        loop d = 0f32 for j < d do
          let z = x[j] - y[j]
          in d + z*z
      let wprod = x_w * y_w
      in loop res' = res for k in (reverse (iota r)) do
           if dist <= radiuses[k]
           then res' with [k] = res'[k] + wprod
           else res'
    -- TODO 2. diff by hand (see also runIterRevAD below for how to proceed after)


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
            -- ^ one weight per querry
          : ([n]i32, [n]i32, [n]f32, [n]i32, [r]f32) =

  let queries_sorted = gather2D queries  query_inds
  let query_ws_sorted= gather1D query_ws query_inds

  -- apply brute force
  let new_res =
    map3 (\ query query_w leaf_ind ->
            if leaf_ind >= i32.i64 num_leaves
            then replicate r 0.0f32
            else bruteForce radiuses query query_w (leaves[leaf_ind]) (ws[leaf_ind])
              -- bruteForce1 radius query query_w res_w leaf_ind leaves ws
         ) queries_sorted query_ws_sorted qleaves
    -- NOTE Big speed-up on primal for this manual rewrite:
    --   |> reduce (map2 (+)) (replicate r 0.0f32) |> opaque
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

  let stacks'  = gather1D new_stacks sort_inds
  let dists'   = gather1D new_dists  sort_inds
  let query_inds' = gather1D query_inds sort_inds

  in  (qleaves', stacks', dists', query_inds', map2 (+) res new_res)

-- def runIterRevAD [q][n][d][num_leaves][ppl][r]
--             (radius: [r]f32)
--             (h: i32)
--             (kd_tree: [q](i32,f32,i32))
--             (leaves:  [num_leaves][ppl][d]f32)
--             (ref_ws:  [num_leaves][ppl]f32)
--             -- ^ invariant
--             (queries: [n][d]f32)
--             (query_ws:[n]f32)
--             -- the loop state:
--             (qleaves:     [n]i32)
--             (stacks:      [n]i32)
--             (dists:       [n]f32)
--             (query_inds:  [n]i32)
--             (res:  f32)
--             -- ^ one weight per querry
--           : ([num_leaves][ppl]f32, [n]f32) =
--   let f (refe_ws , test_ws) : f32 =
--     let (_,_,_,_, res) =
--       iterationSorted radius h kd_tree leaves refe_ws queries test_ws qleaves stacks dists query_inds res
--     in  res
--   in  vjp f (ref_ws, query_ws) 1.0f32
