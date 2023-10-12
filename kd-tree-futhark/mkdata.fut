import "brute-force"
import "util"

entry main (sqrad: f32) (_: i32) (_: [][]f32) (_: []f32) (_: [][]f32) (_: []f32) =
  sqrad

entry brute_primal [m][d][n] (sqrad: f32) (_defppl: i32) (refs: [m][d]f32) (ref_ws: [m]f32) (queries: [n][d]f32) (query_ws: [n]f32) =
    map (\r -> bruteForce r refs ref_ws queries query_ws) (expand_radius 5 sqrad)

entry brute_revad [m][d][n] (sqrad: f32) (_defppl: i32) (refs: [m][d]f32) (ref_ws: [m]f32) (queries: [n][d]f32) (query_ws: [n]f32) : ([5]f32, [5][n]f32, [5][m]f32) =
  map (\r ->
    let f (train_ws, test_ws) = bruteForce r refs train_ws queries test_ws
    let (res, (ref_ws_adj, query_ws_adj)) = vjp2 f (ref_ws, query_ws) 1.0f32
    in  (res, query_ws_adj, ref_ws_adj)
  ) (expand_radius 5 sqrad) |> unzip3
