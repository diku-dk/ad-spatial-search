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

def dupdate [r] (radiuses: [r]f32) dist (res: [r]f32) wprod (resbar0: [r]f32): f32 =
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

  let wprodbar0 = 0f32 -- Free variable that we diff wrt.
  let (_, wprodbar) =
    loop (resbar, wprodbar) = (resbar0, wprodbar0)
    for k in iota r do -- reverse . reverse.
      -- restore
      let res0 = ress[k,k] -- NOTE inefficient storage.
      -- fwd
      let _res1 = if dist <= radiuses[k] then res0 + wprod else res0
      -- rev
      let res1bar = resbar[k]
      let _res0bar = 1 * res1bar -- Both branches are 1.
      let wprodbar' = wprodbar + (if dist <= radiuses[k] then 1 else 0) * res1bar
      in (resbar, wprodbar')
  in wprodbar

-- Doesn't even need res or wprod and arguments.
def dupdate_opt [r] (radiuses: [r]f32) dist (resbar: [r]f32): f32 =
  -- Primal with checkpointing unneeded.
  let wprodbar = 0f32 -- Free variable that we diff wrt.
  let (_, wprodbar) =
    loop (resbar, wprodbar)
    for k in iota r do -- reverse . reverse.
      -- Restore unneeded.
      -- Fwd unneeded.
      -- Rev.
      let wprodbar' = wprodbar + (if dist <= radiuses[k] then 1 else 0) * resbar[k]
      in (resbar, wprodbar')
  in wprodbar

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
    -- TODO 2. diff by hand (see also runIterRevAD below for how to proceed after)

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

  let fbar0 = out_adj
  -- Free variables are x and y.
  let (_fbar, fvsbar) =
    loop (fbar, (xbar_w', ybar_ws')) = (fbar0, (xbar_w, copy ybar_ws))
    for i in reverse (iota m) do
      -- restore
      let res = fs[i]
      -- fwd
      let (y, y_w) = (ys[i], y_ws[i])
      let dist = dist_sq x y
      let wprod = x_w * y_w
      let _res' = update radiuses dist res wprod
      -- rev
      let wprodbar = dupdate radiuses dist res wprod fbar
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
  let (fvsbar) =
    loop (xbar_w, ybar_ws)
    for i in reverse (iota m) do
      -- Restore unneeded.
      -- Fwd.
      let (y, y_w) = (ys[i], y_ws[i])
      let dist = dist_sq x y
      -- Rev.
      let wprodbar = dupdate_opt radiuses dist out_adj
      let xbar_w = xbar_w + y_w * wprodbar
      -- ybar0_ws[i] is zero, so no need to read this value.
      -- let ybar_ws = ybar_ws' with [i] = ybar_ws'[i] + x_w * wprodbar
      let ybar_ws = ybar_ws with [i] = x_w * wprodbar
      in (xbar_w, ybar_ws)
  in fvsbar

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
    let wprodbar = dupdate_opt radiuses dist out_adj
    let xbar_w = y_w * wprodbar
    let ybar_w = ybar_w + x_w * wprodbar
    in (xbar_w, ybar_w)
  ) ys y_ws ybar_ws
  in (xbar_w + reduce (+) 0 xbar_w', ybar_ws)

-- ==
-- compiled input @ data/5radiuses-brute-force-input-refs-512K-queries-1M.out
-- output { true  }
def main [m][d][n][r]
         (radiuses: [r]f32)
         (xs: [n][d]f32) -- One point from sample 1.
         (x_ws: [n]f32)
         (ys: [m][d]f32) -- Sample 2.
         (y_ws: [m]f32) =
  let n = 1000
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
    let expected = vjp f (x_w, y_ws) out_adj
    let got = dbruteForce_opt_soacs radiuses x x_w ys y_ws xbar_w0 ybar_ws0 out_adj
    in expected == got
 ) (iota n) |> reduce (&&) true

-- ==
-- entry: test_update
--
-- compiled input @ data/5radiuses-brute-force-input-refs-512K-queries-1M.out
-- output { true }
entry test_update [m][d][n][r]
                  (radiuses: [r]f32)
                  (_xs: [n][d]f32) -- One point from sample 1.
                  (_x_ws: [n]f32)
                  (_ys: [m][d]f32) -- Sample 2.
                  (_y_ws: [m]f32) =
  let dist = 0.009f32
  let wprod = 3f32
  let f = update radiuses dist (replicate r 0f32)
  let expected = vjp f wprod ((replicate r 0f32) with [0] = 1f32)
  let expected = #[trace(vjp)] expected
  let got = #[trace(manual)] dupdate radiuses dist (replicate r 0f32) wprod ((replicate r 0f32) with [0] = 1f32)
  in expected == got
