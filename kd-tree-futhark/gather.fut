-- futhark c gather.fut && echo "[1i32, 2i32, 3i32, 4i32, 5i32] [1i32, 0i32, 3i32, 2i32, 0i32]" | ./gather
import "util"

def dgather_i32 [m][n] (xsbar: [n]i32) (is: [m]i32) (resbar: [m]i32) : *[n]i32 =
  -- map2 (\i xs' -> xs'[i]) is (replicate n xs)
  reduce_by_index (copy xsbar) (+) 0i32 (map i64.i32 is) resbar

def main [n] (xs: [n]i32) (inds: [n]i32) =
  let (x, y) = vjp2 (flip gather inds) xs xs
  let z = dgather_i32 (replicate n 0i32) inds xs
  in (xs, x, y, z)
  -- vjp (\ys -> gather ys inds) xs xs
