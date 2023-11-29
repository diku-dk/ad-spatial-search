-- ==
-- entry: test_primal
-- compiled input @ data/kdtree-prop-refs-512K-queries-1M.in
-- output @ data/5radiuses-brute-force-primal-refs-512K-queries-1M.out

-- ==
-- entry: test_revad
-- compiled input @ data/kdtree-prop-refs-512K-queries-1M.in
-- output @ data/5radiuses-brute-force-revad-refs-512K-queries-1M.out

-- ==
-- entry: test_revad_by_hand_one_direction test_revad_by_hand test_revad_by_hand_inlined
--
-- compiled input @ data/kdtree-prop-refs-512K-queries-1M.in
-- output { true true }

import "util"
import "map-driver-knn"

entry test_primal = primal

entry test_revad = revad

def is_close t xs ys =
  map2 (\x y -> f32.abs (x - y) <= t) xs ys |> reduce (&&) true

entry test_revad_by_hand_one_direction [d][n][m][m'][q]
        (sq_radius: f32)
        (queries:  [n][d]f32)
        (query_ws: [n]f32)
        (ref_ws:   [m]f32)
        (ref_pts : [m'][d]f32)
        (indir:     [m']i32)
        (median_dims : [q]i32)
        (median_vals : [q]f32)
        (clanc_eqdim : [q]i32) : (bool, bool) =
    -- Using AD.
    let (expected_query_ws', expected_ref_ws') =
      revad_one_direction sq_radius queries query_ws ref_ws ref_pts
                          indir median_dims  median_vals  clanc_eqdim
    -- Manual.
    let (got_query_ws', got_ref_ws') =
      revad_by_hand_one_direction sq_radius queries query_ws ref_ws ref_pts
                                  indir median_dims  median_vals  clanc_eqdim
    -- TODO can we get equality here?
    in (is_close 1e-6 expected_query_ws' got_query_ws',
        is_close 1e-6 expected_ref_ws' got_ref_ws')

entry test_revad_by_hand [d][n][m][m'][q]
        (sq_radius: f32)
        (queries:  [n][d]f32)
        (query_ws: [n]f32)
        (ref_ws:   [m]f32)
        (ref_pts : [m'][d]f32)
        (indir:     [m']i32)
        (median_dims : [q]i32)
        (median_vals : [q]f32)
        (clanc_eqdim : [q]i32) : (bool, bool) =
    -- Using AD.
    let (expected_query_ws', expected_ref_ws') =
      revad sq_radius queries query_ws ref_ws ref_pts
            indir median_dims  median_vals  clanc_eqdim
    -- Manual.
    let (got_query_ws', got_ref_ws') =
      revad_by_hand sq_radius queries query_ws ref_ws ref_pts
                    indir median_dims  median_vals  clanc_eqdim
    -- TODO can we get equality here?
    -- NOTE only 1e-5 on second test
    in (is_close 1e-6 (flatten expected_query_ws') (flatten got_query_ws'),
        is_close 1e-5 (flatten expected_ref_ws') (flatten got_ref_ws'))

entry test_revad_by_hand_inlined [d][n][m][m'][q]
        (sq_radius: f32)
        (queries:  [n][d]f32)
        (query_ws: [n]f32)
        (ref_ws:   [m]f32)
        (ref_pts : [m'][d]f32)
        (indir:     [m']i32)
        (median_dims : [q]i32)
        (median_vals : [q]f32)
        (clanc_eqdim : [q]i32) : (bool, bool) =
    -- Using AD.
    let (expected_query_ws', expected_ref_ws') =
      revad sq_radius queries query_ws ref_ws ref_pts
            indir median_dims  median_vals  clanc_eqdim
    -- Manual.
    let (got_query_ws', got_ref_ws') =
      revad_by_hand_inlined sq_radius queries query_ws ref_ws ref_pts
                            indir median_dims  median_vals  clanc_eqdim
    -- TODO can we get equality here?
    -- NOTE only 1e-5 on second test
    in (is_close 1e-6 (flatten expected_query_ws') (flatten got_query_ws'),
        is_close 1e-5 (flatten expected_ref_ws') (flatten got_ref_ws'))
