import "buildKDtree"
import "knn-iteration"
import "util"
import "kd-traverse"

def propagate [m1][m][q][d][n] (radius: f32)
                           (ref_pts: [m][d]f32)
                           (indir:   [m]i32)
                           (kd_tree: [q](i32,f32,i32))
                           (queries: [n][d]f32)
                           (query_ws:[n]f32, ref_ws_orig: [m1]f32)
                           : f32 =

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
  let res_ws = 0f32

  let (_qleaves', _stacks', _dists', _query_inds', res_ws') =
    loop (qleaves : [n]i32, stacks : [n]i32, dists : [n]f32, query_inds : [n]i32, res_ws : f32)
      for _i < 8 do
        iterationSorted radius h kd_tree leaves kd_ws_sort queries query_ws qleaves stacks dists query_inds res_ws

  in  res_ws'

def rev_prop [m1][m][q][d][n][r]
             (radiuses: [r]f32)
             (ref_pts: [m][d]f32)
             (indir: [m]i32)
             (kd_tree: [q](i32,f32,i32))
             (queries: [n][d]f32)
             -- diff w.r.t weights of kd-tree
             (query_ws:[n]f32, ref_ws_orig: [m1]f32)
             : [r](f32, [n]f32, [m1]f32) =
  map (\radius ->
    let f = propagate radius ref_pts indir kd_tree queries
    let (res, (query_ws_adj, ref_ws_adj)) = vjp2 f (query_ws, ref_ws_orig) 1.0f32
    in (res, query_ws_adj, ref_ws_adj)
  ) radiuses

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
    let rs = replicate_radius sq_radius
    let tree = (zip3 median_dims median_vals clanc_eqdim)
    in map (\sq_radius ->
      propagate sq_radius refs_pts indir tree queries (query_ws, ref_ws)
    ) rs


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
    let rs = replicate_radius sq_radius
    let tree = (zip3 median_dims median_vals clanc_eqdim)
    let (results, query_ws_adjs, ref_ws_adjs) =
      rev_prop rs refs_pts indir tree queries (query_ws, ref_ws)
      |> unzip3
    in (results, query_ws_adjs, ref_ws_adjs)
