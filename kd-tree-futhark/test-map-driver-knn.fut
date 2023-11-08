-- ==
-- entry: primal_test
--
-- compiled input @ data/kdtree-prop-refs-512K-queries-1M.in
-- output @ data/5radiuses-brute-force-primal-refs-512K-queries-1M.out

-- ==
-- entry: revad_test
--
-- compiled input @ data/kdtree-prop-refs-512K-queries-1M.in
-- output @ data/5radiuses-brute-force-revad-refs-512K-queries-1M.out

-- ==
-- entry: revad_by_hand_SINGLE_test revad_by_hand_ALL_test
--
-- compiled input @ data/kdtree-prop-refs-512K-queries-1M.in
-- output { true }

import "util"
import "map-driver-knn"

entry primal_test = primal

entry revad_test = revad

entry revad_by_hand_SINGLE_test [d][n][m][m'][q]
        (sq_radius: f32)
        (queries:  [n][d]f32)
        (query_ws: [n]f32)
        (ref_ws:   [m]f32)
        (refs_pts : [m'][d]f32)
        (indir:     [m']i32)
        (median_dims : [q]i32)
        (median_vals : [q]f32)
        (clanc_eqdim : [q]i32) : bool =
    -- SINGLE DIRECTION:
    let DIR = (replicate 5 1f32)
    let rs = expand_radius 5 sq_radius
    let tree = (zip3 median_dims median_vals clanc_eqdim)

    let f = propagate rs refs_pts indir tree queries
    let (query_ws_adj, _ref_ws_adj) =
      vjp f (query_ws, ref_ws) DIR

    let manual_query_ws_adj =
      diff_propagate rs refs_pts indir tree queries (query_ws, ref_ws) DIR
    in map2 (\x y -> f32.abs (x - y) <= 1e-6) query_ws_adj manual_query_ws_adj
    -- in map2 (==) query_ws_adj manual_query_ws_adj
       |> reduce (&&) true

entry revad_by_hand_ALL_test [d][n][m][m'][q]
        (sq_radius: f32)
        (queries:  [n][d]f32)
        (query_ws: [n]f32)
        (ref_ws:   [m]f32)
        (ref_pts : [m'][d]f32)
        (indir:     [m']i32)
        (median_dims : [q]i32)
        (median_vals : [q]f32)
        (clanc_eqdim : [q]i32) : bool =
    let r = 5
    let rs = expand_radius r sq_radius
    let kd_tree = (zip3 median_dims median_vals clanc_eqdim)

    -- Using AD.
    let (expected_query_ws', expected_ref_ws') =
      let f = propagate rs ref_pts indir kd_tree queries
      in tabulate r (\i ->
        vjp f (query_ws, ref_ws) ((replicate r 0f32) with [i] = 1f32)
      ) |> unzip2

    -- Manual.
    let out_adjs = tabulate r (\i -> (replicate r 0f32) with [i] = 1f32)
    let got_query_ws' =
      diff_propagate_ALL rs ref_pts indir kd_tree queries (query_ws, ref_ws) out_adjs
    in map2 (\x y -> f32.abs (x - y) <= 1e-6)
            (flatten expected_query_ws')
            (flatten got_query_ws')
       |> reduce (&&) true
