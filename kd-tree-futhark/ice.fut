def update [r] (res: *[r]f32) x: [r]f32 =
  replicate r 1f32
  -- loop res' = replicate r 0f32 for k in (iota r) do
  --   res' with [k] = res'[k] + wprod

def main r =
  let f = update (replicate r 0f32)
  in vjp f 3f32 (replicate r 1f32)
