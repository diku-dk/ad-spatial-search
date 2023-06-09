def imap a f = map f a
def imap2 a b f = map2 f a b

def sumSqrsSeq [d] (xs: [d]f32) (ys: [d]f32) : f32 =
    #[sequential]
    map2 (\x y -> let z = x - y in z*z) xs ys
    |> #[sequential]
       reduce (+) 0.0

def bruteForce [m][n][d] (radius: f32) (refs: [m][d]f32) (ref_ws: [m]f32) (queries: [n][d]f32) (query_ws: [n]f32) : f32 =
  imap2 queries (iota n)
  	(\ query i -> -- q_w ->
      #[sequential]
  	  map2
  	  	(\ ref j -> -- r_w ->
  	  		let dsq = sumSqrsSeq query ref
  	  		in  if dsq <= radius
  	  			then ref_ws[j] * query_ws[i] -- q_w * r_w
  	  			else 0.0
  	  	) refs (iota m)
  	  |> #[sequential]
         reduce (+) 0.0f32
  	) 
  |> reduce (+) 0.0f32


-- ==
-- entry: primal revad
--
-- compiled input @ data/sqrad-dot01-leaf-256-refs-512Kx3-dot1-10dot1-ws-dot1-dot2-queries-1M.in

entry primal [m][d][n] (sqrad: f32) (_defppl: i32) (refs: [m][d]f32) (ref_ws: [m]f32) (queries: [n][d]f32) (query_ws: [n]f32) =
    bruteForce sqrad refs ref_ws queries query_ws 

entry revad [m][d][n] (sqrad: f32) (_defppl: i32) (refs: [m][d]f32) (ref_ws: [m]f32) (queries: [n][d]f32) (query_ws: [n]f32) =
  let f (train_ws, test_ws) = bruteForce sqrad refs train_ws queries test_ws
  let (res, (ref_ws_adj, query_ws_adj)) = vjp2 f (ref_ws, query_ws) 1.0f32
  in  (res, query_ws_adj, ref_ws_adj)