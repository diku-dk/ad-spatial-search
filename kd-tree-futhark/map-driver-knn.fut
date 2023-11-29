-- ==
-- entry: primal revad revad_one_direction revad_by_hand_one_direction revad_by_hand revad_by_hand_inlined
--
-- compiled input @ data/kdtree-prop-refs-512K-queries-1M.in

import "buildKDtree"
import "map-knn-iteration"
import "util"
import "kd-traverse"

def setup [m1][m][q][d][n][r]
          (radiuses: [r]f32)
          (ref_pts: [m][d]f32)
          (indir:   [m]i32)
          (kd_tree: [q](i32,f32,i32))
          (queries: [n][d]f32)
          (ref_ws_orig: [m1]f32) =
  -- rearranging the original weights of the reference points
  -- to match the (re-ordered) position in the kd-tree
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

  in (h, leaves, kd_ws_sort, qleaves, query_inds, dists, stacks,
      res_ws, max_radius)

def propagate [m1][m][q][d][n][r]
              (radiuses: [r]f32)
              (ref_pts: [m][d]f32)
              (indir:   [m]i32)
              (kd_tree: [q](i32,f32,i32))
              (queries: [n][d]f32)
              (query_ws:[n]f32, ref_ws_orig: [m1]f32)
              : [r]f32 =
  let (h, leaves, kd_ws_sort, qleaves, query_inds, dists, stacks,
       res_ws, max_radius) =
    setup radiuses ref_pts indir kd_tree queries ref_ws_orig
  let (_qleaves', _stacks', _dists', _query_inds', res_ws') =
    loop (qleaves : [n]i32, stacks : [n]i32, dists : [n]f32, query_inds : [n]i32, res_ws : [r]f32)
      for _i < 8 do
        iterationSorted max_radius radiuses h kd_tree leaves kd_ws_sort queries
                        query_ws qleaves stacks dists query_inds res_ws
  in  res_ws'

def diff_propagate [m1][m][q][d][n][r]
              (radiuses: [r]f32)
              (ref_pts: [m][d]f32)
              (indir:   [m]i32)
              (kd_tree: [q](i32,f32,i32))
              (queries: [n][d]f32)
              (query_ws:[n]f32, ref_ws_orig: [m1]f32)
              (resbar: [r]f32)
              : ([n]f32, [m1]f32) =
  let (h, leaves, kd_ws_sort, qleaves, query_inds, dists, stacks,
       res_ws, max_radius) =
    setup radiuses ref_pts indir kd_tree queries ref_ws_orig
  let query_ws_bar = replicate n 0f32
  let kd_ws_bar = replicate (q + 1) (replicate (m / (q + 1)) 0f32)
  let (_qleaves', _stacks', _dists', _query_inds', _res_ws', query_ws_bar', kd_ws_bar', _resbar) =
    loop (qleaves: [n]i32, stacks: [n]i32, dists: [n]f32, query_inds: [n]i32, res_ws: [r]f32, query_ws_bar, kd_ws_bar, resbar)
      for _i < 8 do
        diterationSorted max_radius radiuses h kd_tree leaves kd_ws_sort queries
                         query_ws qleaves stacks dists query_inds res_ws
                         query_ws_bar kd_ws_bar resbar
  -- Reverse pass of setup (kd_ws was expanded in setup; contract).
  let kd_ws_bar' = sized m (flatten kd_ws_bar')
  let kd_ws_bar' = scatter (replicate m1 0f32) (map i64.i32 indir) kd_ws_bar'
  in (query_ws_bar', kd_ws_bar')

def diff_propagate_ALL [m1][m][q][d][n][r]
                       (radiuses: [r]f32)
                       (ref_pts: [m][d]f32)
                       (indir:   [m]i32)
                       (kd_tree: [q](i32,f32,i32))
                       (queries: [n][d]f32)
                       (query_ws:[n]f32, ref_ws_orig: [m1]f32)
                       (resbars: [r][r]f32)
                       : ([r][n]f32, [r][m1]f32) =
  let (h, leaves, kd_ws_sort, qleaves, query_inds, dists, stacks,
       _res_ws, max_radius) =
    setup radiuses ref_pts indir kd_tree queries ref_ws_orig
  let query_ws_bar = replicate r (replicate n 0f32)
  let kd_ws_bar = replicate r (replicate (q + 1) (replicate (m / (q + 1)) 0f32))
  let res = replicate r 0f32
  let (_qleaves', _stacks', _dists', _query_inds', _res_ws', query_ws_bar', kd_ws_bar', _resbar) =
    loop (qleaves: [n]i32, stacks: [n]i32, dists: [n]f32, query_inds: [n]i32, res, query_ws_bar, kd_ws_bar, resbars)
      for _i < 8 do
        diterationSorted_ALL
          max_radius radiuses h kd_tree leaves kd_ws_sort queries
          query_ws qleaves stacks dists query_inds
          res
          query_ws_bar kd_ws_bar resbars
  -- Reverse pass of setup.
  let kd_ws_bar' = map (\x ->
    let x = sized m (flatten x)
    let x = scatter (replicate m1 0f32) (map i64.i32 indir) x
    in x
  ) kd_ws_bar'
  in (query_ws_bar', kd_ws_bar')

def diff_propagate_ALL_inlined [m1][m][q][d][n][r]
                               (radiuses: [r]f32)
                               (ref_pts: [m][d]f32)
                               (indir:   [m]i32)
                               (kd_tree: [q](i32,f32,i32))
                               (queries: [n][d]f32)
                               (query_ws:[n]f32, ref_ws_orig: [m1]f32)
                               (resbars: [r][r]f32)
                               : ([r][n]f32, [r][m1]f32) =
  let (h, leaves, kd_ws_sort, qleaves, query_inds, dists, stacks,
       _res_ws, max_radius) =
    setup radiuses ref_pts indir kd_tree queries ref_ws_orig
  let query_ws_bar = replicate r (replicate n 0f32)
  let kd_ws_bar = replicate r (replicate (q + 1) (replicate (m / (q + 1)) 0f32))
  let (_qleaves', _stacks', _dists', _query_inds', query_ws_bar', kd_ws_bar', _resbar) =
    loop (qleaves: [n]i32, stacks: [n]i32, dists: [n]f32, query_inds: [n]i32, query_ws_bar, kd_ws_bar, resbars)
      for _i < 8 do
        diterationSorted_ALL_inlined
          max_radius radiuses h kd_tree leaves kd_ws_sort queries
          query_ws qleaves stacks dists query_inds
          query_ws_bar kd_ws_bar resbars
  -- Reverse pass of setup.
  let kd_ws_bar' = map (\x ->
    let x = sized m (flatten x)
    let x = scatter (replicate m1 0f32) (map i64.i32 indir) x
    in x
  ) kd_ws_bar'
  in (query_ws_bar', kd_ws_bar')

entry primal [d][n][m][m'][q]
        (sq_radius: f32)
        (queries:  [n][d]f32)
        (query_ws: [n]f32)
        (ref_ws:   [m]f32)
        (refs_pts : [m'][d]f32)
        (indir:     [m']i32)
        (median_dims : [q]i32)
        (median_vals : [q]f32)
        (clanc_eqdim : [q]i32) : [5]f32 =
    let rs = expand_radius 5 sq_radius
    let tree = (zip3 median_dims median_vals clanc_eqdim)
    in propagate rs refs_pts indir tree queries (query_ws, ref_ws)


entry revad_one_direction [d][n][m][m'][q]
        (sq_radius: f32)
        (queries:  [n][d]f32)
        (query_ws: [n]f32)
        (ref_ws:   [m]f32)
        (refs_pts : [m'][d]f32)
        (indir:     [m']i32)
        (median_dims : [q]i32)
        (median_vals : [q]f32)
        (clanc_eqdim : [q]i32) : ([n]f32, [m]f32) =
    let r = 5
    let dir = (replicate r 0f32) with [0] = 1f32
    let rs = expand_radius r sq_radius
    let tree = (zip3 median_dims median_vals clanc_eqdim)

    let f = propagate rs refs_pts indir tree queries
    in vjp f (query_ws, ref_ws) dir

entry revad [d][n][m][m'][q]
        (sq_radius: f32)
        (queries:  [n][d]f32)
        (query_ws: [n]f32)
        (ref_ws:   [m]f32)
        (refs_pts : [m'][d]f32)
        (indir:     [m']i32)
        (median_dims : [q]i32)
        (median_vals : [q]f32)
        (clanc_eqdim : [q]i32) : ([5][n]f32, [5][m]f32) =
    let r = 5
    let rs = expand_radius r sq_radius
    let tree = (zip3 median_dims median_vals clanc_eqdim)

    let f = propagate rs refs_pts indir tree queries
    in tabulate r (\i ->
      vjp f (query_ws, ref_ws) ((replicate r 0f32) with [i] = 1f32)
    ) |> unzip2

entry revad_by_hand_one_direction [d][n][m][m'][q]
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
    let dir = (replicate r 0f32) with [0] = 1f32
    let tree = (zip3 median_dims median_vals clanc_eqdim)
    in diff_propagate rs refs_pts indir tree queries (query_ws, ref_ws) dir

entry revad_by_hand [d][n][m][m'][q]
        (sq_radius: f32)
        (queries:  [n][d]f32)
        (query_ws: [n]f32)
        (ref_ws:   [m]f32)
        (ref_pts : [m'][d]f32)
        (indir:     [m']i32)
        (median_dims : [q]i32)
        (median_vals : [q]f32)
        (clanc_eqdim : [q]i32) =
    let r = 5
    let rs = expand_radius r sq_radius
    let kd_tree = (zip3 median_dims median_vals clanc_eqdim)
    let out_adjs = tabulate r (\i -> (replicate r 0f32) with [i] = 1f32)
    in diff_propagate_ALL rs ref_pts indir kd_tree queries (query_ws, ref_ws) out_adjs

entry revad_by_hand_inlined [d][n][m][m'][q]
        (sq_radius: f32)
        (queries:  [n][d]f32)
        (query_ws: [n]f32)
        (ref_ws:   [m]f32)
        (ref_pts : [m'][d]f32)
        (indir:     [m']i32)
        (median_dims : [q]i32)
        (median_vals : [q]f32)
        (clanc_eqdim : [q]i32) =
    let r = 5
    let rs = expand_radius r sq_radius
    let kd_tree = (zip3 median_dims median_vals clanc_eqdim)
    let out_adjs = tabulate r (\i -> (replicate r 0f32) with [i] = 1f32)
    in diff_propagate_ALL_inlined rs ref_pts indir kd_tree queries (query_ws, ref_ws) out_adjs

-- Sooo something is making this version slooow
