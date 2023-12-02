def dist_sq [d] (x: [d]f32) (y: [d]f32) =
  loop s = 0f32 for j < d do
    let z = x[j] - y[j]
    in s + z*z

-- TODO fix/report ICE in ice.fut
def update [r] (radiuses: [r]f32) dist (res2: [r]f32) wprod: [r]f32 =
  let res1 =
    loop res = replicate r 0f32 for k in (reverse (iota r)) do
      if dist <= radiuses[k]
      then res with [k] = res[k] + wprod
      else res
  in map2 (+) res1 res2
  -- ^ Alternative to setting `res = res2` to avoid ICE.

-- Doesn't even need res or wprod and arguments.
def dupdate_opt [r] (radiuses: [r]f32) dist (wprodbar: f32) (resbar: [r]f32): f32 =
  -- Primal with checkpointing unneeded.
  let (_, wprodbar) =
    loop (resbar, wprodbar)
    for k in iota r do -- reverse . reverse.
      -- Restore unneeded.
      -- Fwd unneeded.
      -- Rev.
      let wprodbar' = wprodbar + (if dist <= radiuses[k] then 1 else 0) * resbar[k]
      in (resbar, wprodbar')
  in wprodbar

def dupdate_opt_ALL [r] (radiuses: [r]f32) dist (wprodbar: f32) (resbars: [r][r]f32): [r]f32 =
  -- TODO optimise this (could move it inside loop or similar).
  map (dupdate_opt radiuses dist wprodbar) resbars

def bruteForce [m][d][r]
               (radiuses: [r]f32)
               (x: [d]f32) -- One point from sample 1.
               (x_w: f32)
               (ys: [m][d]f32) -- Sample 2.
               (y_ws: [m]f32)
               : [r]f32 =
    loop res = replicate r 0f32 for i < m do
      let (y, y_w) = (ys[i], y_ws[i])
      let dist = dist_sq x y
      let wprod = x_w * y_w
      in update radiuses dist res wprod

def dbruteForce_opt_seq_ALL [m][d][r]
                            (radiuses: [r]f32)
                            (x: [d]f32) -- One point from sample 1.
                            (x_w: f32)
                            (ys: [m][d]f32) -- Sample 2.
                            (y_ws: [m]f32)
                            (xbar_ws: *[r]f32)
                            (ybar_wss: *[r][m]f32)
                            (out_adjs: [r][r]f32)
                            : ([r]f32, [r][m]f32) =
  -- Primal with checkpointing unneeded.
  -- Differentiate w.r.t. free variables x_w and y_ws.
  let (xbar_ws, ybar_wssT) =
    -- NOTE trasnspose ybar_wss to avoid copying inside loop (doing
    -- that results in unsliceable allocation compiler limitation).
    -- See git history for non-transposed version.
    loop (xbar_ws, ybar_wssT) = (xbar_ws, transpose (copy ybar_wss)) -- TODO why does copying here speed things up by 2-3x??
    -- NOTE copying ybar_wss above is beneficial to diff_iteration; by
    -- inspecting the IR, it hoists the replicate that makes ybar_wss
    -- outside the map in diff_iteration. Without that copy, the replicate
    -- is inside the map and further, it also transposes afterwars (don't
    -- know if that's optimised away later, though).
    -- TODO next: try to output transposed ybar_wssT and use that in diff_iteration;
    -- transposing twice there may be more efficient?
    for i in reverse (iota m) do
      -- Restore unneeded.
      -- Fwd.
      let (y, y_w) = (ys[i], y_ws[i])
      let dist = dist_sq x y
      -- Rev.
      let wprodbars = dupdate_opt_ALL radiuses dist 0 out_adjs
      let xbar_ws = map2 (\xbar_w wprodbar -> xbar_w + y_w * wprodbar) xbar_ws wprodbars
      let column_upd = map2 (\ybar_w wprodbar -> ybar_w + x_w * wprodbar)
                            (ybar_wssT[i]) wprodbars
      let ybar_wssT = ybar_wssT with [i] = column_upd
      in (xbar_ws, ybar_wssT)
  let fvsbar = (xbar_ws, transpose ybar_wssT)
  in fvsbar

def dbruteForce_opt_seq_ALL_T [m][d][r]
                            (radiuses: [r]f32)
                            (x: [d]f32) -- One point from sample 1.
                            (x_w: f32)
                            (ys: [m][d]f32) -- Sample 2.
                            (y_ws: [m]f32)
                            (xbar_ws: [r]f32)
                            (ybar_wssT: [m][r]f32)
                            (out_adjs: [r][r]f32)
                            : ([r]f32, [m][r]f32) =
  -- Primal with checkpointing unneeded.
  -- Differentiate w.r.t. free variables x_w and y_ws.
  let fvsbar =
    loop (xbar_ws, ybar_wssT) = (xbar_ws, (copy ybar_wssT)) -- TODO why does copying here speed things up by 2-3x??
    for i in reverse (iota m) do
      -- Restore unneeded.
      -- Fwd.
      let (y, y_w) = (ys[i], y_ws[i])
      let dist = dist_sq x y
      -- Rev.
      let wprodbars = dupdate_opt_ALL radiuses dist 0 out_adjs
      let xbar_ws = map2 (\xbar_w wprodbar -> xbar_w + y_w * wprodbar) xbar_ws wprodbars
      let column_upd = map2 (\ybar_w wprodbar -> ybar_w + x_w * wprodbar)
                            (ybar_wssT[i]) wprodbars
      let ybar_wssT = ybar_wssT with [i] = column_upd
      in (xbar_ws, ybar_wssT)
  in fvsbar

def dbruteForce_opt_soacs_ALL_T [m][d][r]
                          (radiuses: [r]f32)
                          (x: [d]f32) -- One point from sample 1.
                          (x_w: f32)
                          (ys: [m][d]f32) -- Sample 2.
                          (y_ws: [m]f32)
                          (xbar_ws: *[r]f32)
                          (ybar_wssT: *[m][r]f32)
                          (out_adjs: [r][r]f32)
                          : ([r]f32, [m][r]f32) =
  -- Primal with checkpointing unneeded.
  -- Differentiate w.r.t. free variables x_w and y_ws.
  let (xbar_wss, ybar_wssT) = unzip <| map2 (\i ybar_wsT ->
    -- Restore unneeded.
    -- Fwd.
    let (y, y_w) = (ys[i], y_ws[i])
    let dist = dist_sq x y
    -- Rev.
    let wprodbars = dupdate_opt_ALL radiuses dist 0 out_adjs
    let xbar_ws = map2 (\xbar_w wprodbar -> xbar_w + y_w * wprodbar)
                       xbar_ws wprodbars
    let ybar_wsT = map2 (\ybar_w wprodbar -> ybar_w + x_w * wprodbar)
                        ybar_wsT wprodbars
    in (xbar_ws, ybar_wsT)
  ) (iota m) ybar_wssT
  in (map (reduce (+) 0) (transpose xbar_wss), ybar_wssT)

-- ==
-- entry: bench_manual bench_ad
-- compiled input @ data/5radiuses-brute-force-input-refs-512K-queries-1M.out
entry bench_manual [m][d][n][r]
         (radiuses: [r]f32)
         (xs: [n][d]f32) -- One point from sample 1.
         (x_ws: [n]f32)
         (ys: [m][d]f32) -- Sample 2.
         (y_ws: [m]f32) =
  let i = 0
  let (x, x_w) = (xs[i], x_ws[i])
  let out_adjs = tabulate r (\i -> (replicate r 0f32) with [i] = 1f32)
  let xbar_w0 = replicate r 0f32
  let ybar_ws0 = replicate r (replicate m 0f32)
  in dbruteForce_opt_seq_ALL radiuses x x_w ys y_ws xbar_w0 ybar_ws0 out_adjs

-- entry bench_ad [m][d][n][r]
--          (radiuses: [r]f32)
--          (xs: [n][d]f32) -- One point from sample 1.
--          (x_ws: [n]f32)
--          (ys: [m][d]f32) -- Sample 2.
--          (y_ws: [m]f32) =
--   let i = 0
--   let (x, x_w) = (xs[i], x_ws[i])
--   let f (x_w, y_ws) = bruteForce radiuses x x_w ys y_ws
--   let out_adjs = tabulate r (\i -> (replicate r 0f32) with [i] = 1f32)
--   in unzip <| map (vjp f (x_w, y_ws)) out_adjs
