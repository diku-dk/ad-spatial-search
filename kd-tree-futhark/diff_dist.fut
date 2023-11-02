def f [d] (x: [d]f32, y: [d]f32) =
  loop s = 0f32 for j < d do
    let z = x[j] - y[j]
    in s + z*z

-- def df [d] (x: [d]f32) (y: [d]f32): ([d]f32, [d]f32)=
-- def df (x: [3]f32, y: [3]f32): ([3]f32, [3]f32)=
--   -- let primal = f(x, y)
--   let z0 = x[0] - y[0]
--   let z1 = x[1] - y[1]
--   let z2 = x[2] - y[2]
--   let f = z0*z0 + z1*z1 + z2*z2

--   -- loop s = 0f32 for j < d do
--   --   let z = x[j] - y[j]
--   --   in s + z*z

--   let fbar = 1f32

--   let z0bar = 2*z0 * fbar -- d f / dz0 * fbar
--   let z1bar = 2*z1 * fbar
--   let z2bar = 2*z2 * fbar
--   let x2bar =  1 * z2bar
--   let y2bar = -1 * z2bar
--   let x1bar =  1 * z1bar
--   let y1bar = -1 * z1bar
--   let x0bar =  1 * z0bar
--   let y0bar = -1 * z0bar
--   in ([x0bar, x1bar, x2bar], [y0bar, y1bar, y2bar])
def df [d] (x: [d]f32, y: [d]f32): ([d]f32, [d]f32)=
  -- Primal with checkpointing.
  let (_f, fs) =
    loop (f', fs') = (0f32, replicate d 0f32)
    for j < d do
      let z = x[j] - y[j]
      in (f' + z*z, fs' with [j] = f' + z*z)

  let fbar0 = 1
  -- Free variables are x and y.
  let xbar0 = replicate d 0f32
  let ybar0 = replicate d 0f32
  let (_fbar, fvsbar) =
    loop (fbar, (xbar, ybar)) = (fbar0, (xbar0, ybar0))
    for j in reverse (iota d) do
      -- restore
      let f0 = fs[j]
      -- fwd
      let z = x[j] - y[j]
      let _f = f0 + z*z
      -- rev
      let f0bar = 1 * fbar
      let zbar = 2*z * fbar
      let xbar_j = 1 * zbar
      let ybar_j = -1 * zbar
      in (f0bar, (xbar with [j] = xbar_j, ybar with [j] = ybar_j))
  in fvsbar
  -- NOTE^ Note that primal is unused except for `z` here.
  --       In particular, the array expansion in the primal
  --       is unnecessary since the loop is a fold.

def df_opt [d] (x: [d]f32, y: [d]f32): ([d]f32, [d]f32)=
  let zs = map2 (-) x y
  let qs = map2 (*) zs zs
  let f = reduce (+) 0f32 qs

  let fbar = 1
  -- let qbars = replicate d 0f32
  -- let qbars = replicate d fbar |> map2 (+) qbars
  -- let zbars = map2 (\z qbar -> 2*z * qbar) zs qbars -- qbars = zeros
  let zbars = map (\z -> 2*z * fbar) zs -- Equal to the above three lines.
  let xbars = map2 (\_x zbar -> 1*zbar) x zbars -- xbars = zeros
  let ybars = map2 (\_y zbar -> -1*zbar) y zbars -- ybars = zeros
  in (xbars, ybars)

def df_opt_seq [d] (x: [d]f32, y: [d]f32): ([d]f32, [d]f32)=
  -- NOTE no primal needed!
  let fbar0 = 1
  -- Free variables are x and y.
  let xbar0 = replicate d 0f32
  let ybar0 = replicate d 0f32
  let (_fbar, fvsbar) =
    loop (fbar, (xbar, ybar)) = (fbar0, (xbar0, ybar0))
    for j < d do
      let z = x[j] - y[j]
      -- rev
      let f0bar = 1 * fbar
      let zbar = 2*z * fbar
      let xbar_j = 1 * zbar
      let ybar_j = -1 * zbar
      in (f0bar, (xbar with [j] = xbar_j, ybar with [j] = ybar_j))
  in fvsbar

-- ==
-- input { [1f32,2f32,3f32] [4f32,5f32,6f32] }
-- output { true true true }
-- random input { [10]f32 [10]f32 }
-- output { true true true }
-- random input { [100]f32 [100]f32 }
-- output { true true true }
-- random input { [1000]f32 [1000]f32 }
-- output { true true true }
def main x y =
  let inp = (x,y)
  let e = #[trace(vjp)] vjp f inp 1f32
  let a = #[trace(df)] df inp
  let b = #[trace(df_opt)] df_opt inp
  let c = #[trace(df_opt_seq)] df_opt_seq inp
  in (e == a, e == b, e == c)
