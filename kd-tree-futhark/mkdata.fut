import "brute-force"
import "util"

entry main (sqrad: f32) (_: i32) (_: [][]f32) (_: []f32) (_: [][]f32) (_: []f32) =
  sqrad

entry bruteForce_input [m][d][n] (sqrad: f32) (_defppl: i32) (refs: [m][d]f32) (ref_ws: [m]f32) (queries: [n][d]f32) (query_ws: [n]f32) =
  (expand_radius 5 sqrad, queries, query_ws, refs, ref_ws)

entry brute_primal [m][d][n] (sqrad: f32) (_defppl: i32) (refs: [m][d]f32) (ref_ws: [m]f32) (queries: [n][d]f32) (query_ws: [n]f32) =
    map (\r -> bruteForce r refs ref_ws queries query_ws) (expand_radius 5 sqrad)

entry brute_revad [m][d][n] (sqrad: f32) (_defppl: i32) (refs: [m][d]f32) (ref_ws: [m]f32) (queries: [n][d]f32) (query_ws: [n]f32) : ([5]f32, [5][n]f32, [5][m]f32) =
  map (\r ->
    let f (train_ws, test_ws) = bruteForce r refs train_ws queries test_ws
    let (res, (ref_ws_adj, query_ws_adj)) = vjp2 f (ref_ws, query_ws) 1.0f32
    in  (res, query_ws_adj, ref_ws_adj)
  ) (expand_radius 5 sqrad) |> unzip3

import "kd-traverse"
import "map-knn-iteration" -- This redefines bruteForce from the above.

entry iterationSorted_input [d][n][m][m'][q]
        (sq_radius: f32)
        (queries:  [n][d]f32)
        (query_ws: [n]f32)
        (ref_ws:   [m]f32)
        (refs_pts : [m'][d]f32)
        (indir:     [m']i32)
        (median_dims : [q]i32)
        (median_vals : [q]f32)
        (clanc_eqdim : [q]i32) =
    let r = 5
    let rs = expand_radius r sq_radius
    let tree = (zip3 median_dims median_vals clanc_eqdim)

    let radiuses = rs
    let ref_pts = refs_pts
    let kd_tree = tree
    let ref_ws_orig = ref_ws
    let m1 = m

    let kd_weights =
          map i64.i32 indir |>
          map (\ind -> if ind >= m1 then 1.0f32 else ref_ws_orig[ind])

    let (median_dims, median_vals, _) = unzip3 kd_tree
    let num_nodes  = q -- trace q
    let num_leaves = num_nodes + 1
    let h = (log2 (i32.i64 num_leaves)) - 1
    let ppl = m / num_leaves
    let leaves = unflatten (sized (num_leaves*ppl) ref_pts)
    let kd_ws_sort = unflatten (sized (num_leaves*ppl) kd_weights)

    let query_leaves = map (findLeaf median_dims median_vals h) queries
    let (qleaves, query_inds) = sortQueriesByLeavesRadix (h+1) query_leaves
    let dists  = replicate n 0.0f32
    let stacks = replicate n 0i32
    let res_ws = replicate r 0f32

    -- Max radius is used in traversal decisions.
    let max_radius = f32.maximum radiuses
    -- kd_tree is (median_dims, median_vals, clanc_eqdim)
    in (max_radius, radiuses, h, median_dims, median_vals, clanc_eqdim, leaves, kd_ws_sort, queries, query_ws, qleaves, stacks, dists, query_inds, res_ws)

