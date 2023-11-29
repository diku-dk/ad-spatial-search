def dist_sq [d] (x: [d]f32) (y: [d]f32) =
  loop s = 0f32 for j < d do
    let z = x[j] - y[j]
    in s + z*z

-- See diff_dist.fut.
-- Not neeeded.
-- def ddist_sq [d] (x: [d]f32, y: [d]f32): ([d]f32, [d]f32)=
--   let fbar0 = 1
--   -- Free variables are x and y.
--   let xbar0 = replicate d 0f32
--   let ybar0 = replicate d 0f32
--   let (_fbar, fvsbar) =
--     loop (fbar, (xbar, ybar)) = (fbar0, (xbar0, ybar0))
--     for j < d do
--       let z = x[j] - y[j]
--       -- rev
--       let f0bar = 1 * fbar
--       let zbar = 2*z * fbar
--       let xbar_j = 1 * zbar
--       let ybar_j = -1 * zbar
--       in (f0bar, (xbar with [j] = xbar_j, ybar with [j] = ybar_j))
--   in fvsbar

-- TODO fix/report ICE in ice.fut
def update [r] (radiuses: [r]f32) dist (res2: [r]f32) wprod: [r]f32 =
  let res1 =
    loop res = replicate r 0f32 for k in (reverse (iota r)) do
      if dist <= radiuses[k]
      then res with [k] = res[k] + wprod
      else res
  in map2 (+) res1 res2
  -- ^ Alternative to setting `res = res2` to avoid ICE.

def dupdate [r] (radiuses: [r]f32) dist (res: [r]f32) wprod (wprodbar0: f32) (resbar0: [r]f32): ([r]f32, f32) =
  -- NOTE no information is destroyed in the primal (scan-like).
  -- So the reverse pass can be optimised.
  let res = copy res
  let (_res, ress) =
    loop (res', ress') = (res, replicate r res)
    for k in (reverse (iota r)) do
      if dist <= radiuses[k]
      then let upd = res'[k] + wprod
           in (res' with [k] = upd, ress' with [k,k] = upd)
      else (res', ress')

  let (resbar, wprodbar) =
    loop (resbar, wprodbar) = (copy resbar0, wprodbar0)
    for k in iota r do -- reverse . reverse.
      -- restore
      let res0 = ress[k,k] -- NOTE inefficient storage.
      -- fwd
      let _res1 = if dist <= radiuses[k] then res0 + wprod else res0
      -- rev
      let res1bar = resbar[k]
      let res0bar = 1 * res1bar -- Both branches are 1.
      let wprodbar' = wprodbar + (if dist <= radiuses[k] then 1 else 0) * res1bar
      in (resbar with [k] = res0bar, wprodbar')
  in (resbar, wprodbar)

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
    -- map2(\y y_w ->
    --       let dist = sumSqrsSeq x y
    --       in  map (\radius -> if dist <= radius then x_w * y_w else 0.0f32) radiuses
    --     ) ys y_ws
    --- |> reduce (map2 (+)) (replicate r 0.0f32)
    loop res = replicate r 0f32 for i < m do
      let (y, y_w) = (ys[i], y_ws[i])
      let dist = dist_sq x y
      let wprod = x_w * y_w
      in update radiuses dist res wprod
      -- in loop res' = res for k in (reverse (iota r)) do
      --      if dist <= radiuses[k]
      --      then res' with [k] = res'[k] + wprod
      --      else res'

def dbruteForce [m][d][r]
                (radiuses: [r]f32)
                (x: [d]f32) -- One point from sample 1.
                (x_w: f32)
                (ys: [m][d]f32) -- Sample 2.
                (y_ws: [m]f32)
                (xbar_w: f32)
                (ybar_ws: [m]f32)
                (out_adj: [r]f32)
                : (f32, [m]f32) =
  -- let f (x_w, y_ws) = bruteForce radiuses x x_w ys y_ws
  -- in vjp f (x_w, y_ws) out_adj
  -- Primal with checkpointing.
  let (_f, fs) =
    loop (res, ress) = (replicate r 0f32, replicate m (replicate r 0f32))
    for i < m do
      let (y, y_w) = (ys[i], y_ws[i])
      let dist = dist_sq x y
      let wprod = x_w * y_w
      let res' = update radiuses dist res wprod
      in (res', ress with [i] = res')

  -- Free variables are x and y.
  let (_fbar, fvsbar) =
    loop (fbar, (xbar_w', ybar_ws')) = (out_adj, (xbar_w, copy ybar_ws))
    for i in reverse (iota m) do
      -- restore
      let res = fs[i]
      -- fwd
      let (y, y_w) = (ys[i], y_ws[i])
      let dist = dist_sq x y
      let wprod = x_w * y_w
      let _res' = update radiuses dist res wprod
      -- rev
      let wprodbar = 0f32
      let (fbar, wprodbar) = dupdate radiuses dist res wprod wprodbar fbar
      let xbar_w = xbar_w' + y_w * wprodbar
      let ybar_ws = ybar_ws' with [i] = ybar_ws'[i] + x_w * wprodbar
      -- NOTE differentiating dist_sq is unnecessary; the dist
      -- is only used in update for control-flow (i.e., only primal needed).
      -- let (_xbar, _ybar) = ddist_sq (x, y)
      in (fbar, (xbar_w, ybar_ws))
  in fvsbar

def dbruteForce_opt_seq [m][d][r]
                        (radiuses: [r]f32)
                        (x: [d]f32) -- One point from sample 1.
                        (x_w: f32)
                        (ys: [m][d]f32) -- Sample 2.
                        (y_ws: [m]f32)
                        (xbar_w: f32)
                        (ybar_ws: [m]f32)
                        (out_adj: [r]f32)
                        : (f32, [m]f32) =
  -- Primal with checkpointing unneeded.
  -- Differentiate w.r.t. free variables x_w and y_ws.
  let ybar_ws = copy ybar_ws
  let fvsbar =
    loop (xbar_w, ybar_ws)
    for i in reverse (iota m) do
      -- Restore unneeded.
      -- Fwd.
      let (y, y_w) = (ys[i], y_ws[i])
      let dist = dist_sq x y
      -- Rev.
      let wprodbar = dupdate_opt radiuses dist 0 out_adj
      let xbar_w = xbar_w + y_w * wprodbar
      -- ybar0_ws[i] is zero, so no need to read this value below?
      let ybar_ws = ybar_ws with [i] = ybar_ws[i] + x_w * wprodbar
      in (xbar_w, ybar_ws)
  in fvsbar

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
      -- -- Rev.
      -- -- let wprodbars = dupdate_opt_ALL radiuses dist 0 out_adjs
      -- let (wprodbars, xbar_ws) = map2 (\resbar xbar_w ->
      --   let (_, wprodbar) =
      --     loop (resbar, wprodbar) = (resbar, 0)
      --     for k in iota r do -- reverse . reverse.
      --       -- Restore unneeded.
      --       -- Fwd unneeded.
      --       -- Rev.
      --       let wprodbar' = wprodbar + (if dist <= radiuses[k] then 1 else 0) * resbar[k]
      --       in (resbar, wprodbar')
      --   let xbar_w = xbar_w + y_w * wprodbar
      --   in (wprodbar, xbar_w)
      -- ) out_adjs xbar_ws
      -- |> unzip
      -- -- let xbar_ws = map2 (\xbar_w wprodbar -> xbar_w + y_w * wprodbar) xbar_ws wprodbars
      -- let column_upd = map2 (\ybar_w wprodbar -> ybar_w + x_w * wprodbar)
      --                       (ybar_wssT[i]) wprodbars
      -- let ybar_wssT = ybar_wssT with [i] = column_upd

def dbruteForce_opt_soacs [m][d][r]
                          (radiuses: [r]f32)
                          (x: [d]f32) -- One point from sample 1.
                          (x_w: f32)
                          (ys: [m][d]f32) -- Sample 2.
                          (y_ws: [m]f32)
                          (xbar_w: f32)
                          (ybar_ws: [m]f32)
                          (out_adj: [r]f32)
                          : (f32, [m]f32) =
  -- Primal with checkpointing unneeded.
  -- Differentiate w.r.t. free variables x_w and y_ws.
  let (xbar_w', ybar_ws) = unzip <| map3 (\y y_w ybar_w ->
    let dist = dist_sq x y
    -- Rev.
    let wprodbar = dupdate_opt radiuses dist 0 out_adj
    let xbar_w = y_w * wprodbar
    let ybar_w = ybar_w + x_w * wprodbar
    in (xbar_w, ybar_w)
  ) ys y_ws ybar_ws
  in (xbar_w + reduce (+) 0 xbar_w', ybar_ws)

-- ==
-- entry: main main_ALL
-- nobench compiled input @ data/5radiuses-brute-force-input-refs-512K-queries-1M.out
-- output { true true }
def main [m][d][n][r]
         (radiuses: [r]f32)
         (xs: [n][d]f32) -- One point from sample 1.
         (x_ws: [n]f32)
         (ys: [m][d]f32) -- Sample 2.
         (y_ws: [m]f32) =
  let n = 100000
  let m = 10000
  let xs = xs[:n]
  let x_ws = x_ws[:n]
  let ys = ys[:m]
  let y_ws = y_ws[:m]

  in map (\i ->
    let (x, x_w) = (xs[i], x_ws[i])
    let f (x_w, y_ws) = bruteForce radiuses x x_w ys y_ws
    let out_adj = replicate r 1f32
    let xbar_w0 = 0f32
    let ybar_ws0 = replicate m 0f32
    let (expected_x, expected_y) = vjp f (x_w, y_ws) out_adj
    let (got_x, got_y) = dbruteForce radiuses x x_w ys y_ws xbar_w0 ybar_ws0 out_adj
    -- let diffs = filter (uncurry (!=)) (zip expected_x got_x)
    -- let expected _x= #[trace(expected_x)] expected_x
    -- let got_x = #[trace(got_x)] got_x
    in (expected_x == got_x, expected_y == got_y)
 ) (iota n) |> reduce (\(x,x') (y,y') -> (x && x', y && y')) (true, true)

entry main_ALL [m][d][n][r]
         (radiuses: [r]f32)
         (xs: [n][d]f32) -- One point from sample 1.
         (x_ws: [n]f32)
         (ys: [m][d]f32) -- Sample 2.
         (y_ws: [m]f32) =
  let n = 100000
  let m = 10000
  let xs = xs[:n]
  let x_ws = x_ws[:n]
  let ys = ys[:m]
  let y_ws = y_ws[:m]

  in map (\i ->
    let (x, x_w) = (xs[i], x_ws[i])
    let f (x_w, y_ws) = bruteForce radiuses x x_w ys y_ws
    let out_adjs = tabulate r (\i -> (replicate r 0f32) with [i] = 1f32)
    let xbar_w0 = replicate r 0f32
    let ybar_ws0 = replicate r (replicate m 0f32)
    let (expected_x, expected_y) = unzip <| map (vjp f (x_w, y_ws)) out_adjs
    let (got_x, got_y) = dbruteForce_opt_seq_ALL radiuses x x_w ys y_ws xbar_w0 ybar_ws0 out_adjs
    -- let diffs = filter (uncurry (!=)) (zip expected_x got_x)
    -- let expected _x= #[trace(expected_x)] expected_x
    -- let got_x = #[trace(got_x)] got_x
    in (expected_x == got_x, expected_y == got_y)
 ) (iota n) |> reduce (\(x,x') (y,y') -> (x && x', y && y')) (true, true)

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

entry bench_ad [m][d][n][r]
         (radiuses: [r]f32)
         (xs: [n][d]f32) -- One point from sample 1.
         (x_ws: [n]f32)
         (ys: [m][d]f32) -- Sample 2.
         (y_ws: [m]f32) =
  let i = 0
  let (x, x_w) = (xs[i], x_ws[i])
  let f (x_w, y_ws) = bruteForce radiuses x x_w ys y_ws
  let out_adjs = tabulate r (\i -> (replicate r 0f32) with [i] = 1f32)
  in unzip <| map (vjp f (x_w, y_ws)) out_adjs

-- -- ==
-- -- entry: test_update
-- --
-- -- compiled input @ data/5radiuses-brute-force-input-refs-512K-queries-1M.out
-- -- output { true }
-- entry test_update [m][d][n][r]
--                   (radiuses: [r]f32)
--                   (_xs: [n][d]f32) -- One point from sample 1.
--                   (_x_ws: [n]f32)
--                   (_ys: [m][d]f32) -- Sample 2.
--                   (_y_ws: [m]f32) =
--   let dist = 0.009f32
--   let wprod = 3f32
--   let f = update radiuses dist (replicate r 0f32)
--   let expected = vjp f wprod ((replicate r 0f32) with [0] = 1f32)
--   let expected = #[trace(vjp)] expected
--   let got = #[trace(manual)] dupdate radiuses dist (replicate r 0f32) wprod 0f32 ((replicate r 0f32) with [0] = 1f32)
--   in expected == got
