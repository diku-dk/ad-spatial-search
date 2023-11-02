import "buildKDtree"
import "map-knn-iteration"
import "util"
import "kd-traverse"

def propagate [m1][m][q][d][n][r]
              (radiuses: [r]f32)
              (ref_pts: [m][d]f32)
              (indir:   [m]i32)
              (kd_tree: [q](i32,f32,i32))
              (queries: [n][d]f32)
              (query_ws:[n]f32, ref_ws_orig: [m1]f32)
              : [r]f32 =
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

  let (_qleaves', _stacks', _dists', _query_inds', res_ws') =
    loop (qleaves : [n]i32, stacks : [n]i32, dists : [n]f32, query_inds : [n]i32, res_ws : [r]f32)
      for _i < 8 do
        iterationSorted max_radius radiuses h kd_tree leaves kd_ws_sort queries
                        query_ws qleaves stacks dists query_inds res_ws

  in  res_ws'

def rev_prop [m1][m][q][d][n][r]
             (radiuses: [r]f32)
             (ref_pts: [m][d]f32)
             (indir: [m]i32)
             (kd_tree: [q](i32,f32,i32))
             (queries: [n][d]f32)
             -- diff w.r.t weights of kd-tree
             (query_ws:[n]f32, ref_ws_orig: [m1]f32)
             : ([r]f32, [r][n]f32, [r][m1]f32) =
  let f = propagate radiuses ref_pts indir kd_tree queries
  -- TODO We only need res from one of the vjp2s (it's the same across i).
  -- For now, this seems faster than doing vjp2 inside loop:
  let (res) = f (query_ws, ref_ws_orig)
  let (query_ws_adj, ref_ws_adj) = tabulate r (\i ->
    let e = (replicate r 0f32) with [i] = 1f32
    in vjp f (query_ws, ref_ws_orig) e
  ) |> unzip2
  in (res, query_ws_adj, ref_ws_adj)

-- ==
-- entry: primal

--
-- compiled input @ data/kdtree-prop-refs-512K-queries-1M.in
-- output @ data/5radiuses-brute-force-primal-refs-512K-queries-1M.out
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


-- ==
-- entry: revad

--
-- compiled input @ data/kdtree-prop-refs-512K-queries-1M.in
-- output @ data/5radiuses-brute-force-revad-refs-512K-queries-1M.out
entry revad [d][n][m][m'][q]
        (sq_radius: f32)
        (queries:  [n][d]f32)
        (query_ws: [n]f32)
        (ref_ws:   [m]f32)
        (refs_pts : [m'][d]f32)
        (indir:     [m']i32)
        (median_dims : [q]i32)
        (median_vals : [q]f32)
        (clanc_eqdim : [q]i32) : ([5]f32, [5][n]f32, [5][m]f32) =
    let rs = expand_radius 5 sq_radius
    let tree = (zip3 median_dims median_vals clanc_eqdim)
    in rev_prop rs refs_pts indir tree queries (query_ws, ref_ws)


def diff_propagate [m1][m][q][d][n][r]
              (radiuses: [r]f32)
              (ref_pts: [m][d]f32)
              (indir:   [m]i32)
              (kd_tree: [q](i32,f32,i32))
              (queries: [n][d]f32)
              (query_ws:[n]f32, ref_ws_orig: [m1]f32)
              (resbar: [r]f32)
              : [n]f32 =
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

  -- TODO diff this loop.
  -- TODO Validate the results for a single vector resbar against vjp.
  --      That is, result type is [n].
  -- TODO figure out how to get [r][n] and not [n] in one go!
  -- Will require moving resbar all the way in, I think.
  -- There's probably a point at which I'm looping over each element,
  -- without reducing after, and then each element might as well be 1.

  let qleavess = replicate 8 qleaves
  let stackss = replicate 8 stacks
  let distss = replicate 8 dists
  let query_indss = replicate 8 query_inds
  let res_wss = replicate 8 res_ws

  -- let (_qleaves', _stacks', _dists', _query_inds', res_ws',
  --      _qleavess, _stackss, _distss, _query_indss, res_wss) =
  --   loop (qleaves : [n]i32, stacks : [n]i32, dists : [n]f32, query_inds : [n]i32, res_ws : [r]f32,
  --         qleavess, stackss, distss, query_indss, res_wss)
  --     for _i < 8 do
  --       iterationSorted max_radius radiuses h kd_tree leaves kd_ws_sort queries
  --                       query_ws qleaves stacks dists query_inds res_ws

  -- let query_ws_adj = replicate r (replicate n 0f32)
  let query_ws_bar = replicate n 0f32
  let ws_bar = replicate num_leaves (replicate ppl 0f32)
  let (_qleaves', _stacks', _dists', _query_inds', _res_ws', query_ws_bar') =
    loop (qleaves: [n]i32, stacks: [n]i32, dists: [n]f32, query_inds: [n]i32, res_ws: [r]f32, query_ws_bar)
      for _i < 8 do
        let (x0, x1, x2, x3, x4, x5) =
          diff_iterationSorted max_radius radiuses h kd_tree leaves kd_ws_sort queries
                               query_ws qleaves stacks dists query_inds res_ws
                               query_ws_bar ws_bar resbar
        in (x0, x1, x2, x3, x4, x5)

  in query_ws_bar'

-- ==
-- entry: derivative_by_hand
--
-- compiled input @ data/kdtree-prop-refs-512K-queries-1M.in
-- output { true }
entry derivative_by_hand [d][n][m][m'][q]
        (sq_radius: f32)
        (queries:  [n][d]f32)
        (query_ws: [n]f32)
        (ref_ws:   [m]f32)
        (refs_pts : [m'][d]f32)
        (indir:     [m']i32)
        (median_dims : [q]i32)
        (median_vals : [q]f32)
        (clanc_eqdim : [q]i32) : bool =
    -- let (_res, query_ws_adj, _ref_ws_adj) =
      -- revad sq_radius queries query_ws ref_ws refs_pts indir median_dims median_vals clanc_eqdim

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

    -- ALL DIRECTIONS:
    -- in map2 (==) (flatten query_ws_adj) (flatten manual_query_ws_adj)
    --    |> reduce (&&) true
