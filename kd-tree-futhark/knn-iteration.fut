import "lib/github.com/diku-dk/sorts/radix_sort"

import "kd-traverse"
import "util"

def sumSqrsSeq [d] (xs: [d]f32) (ys: [d]f32) : f32 =
    #[sequential]
    map2 (\x y -> let z = x - y in z*z) xs ys
    |> reduce (+) 0.0

--    loop (res) = (0.0f32) for (x,y) in (zip xs ys) do
--        let z = x-y in res + z*z

def seqop (p: f32, s:f32) (w: f32) : (f32, f32) = (p+s*w, s+w)
def parop (p1: f32, s1: f32) (p2: f32, s2: f32) : (f32, f32) =
  (p1 + s1*s2 + p2, s1 + s2)

def bruteForce [m][d] (radius: f32)
                      (query: [d]f32)
                      (query_w:  f32)
                      (leaf_refs : [m][d]f32)
                      (leaf_ws :   [m]f32)
                    : f32 =
--  loop res = 0.0f32
--  for i < m do
--    let dist =
--      loop (dst) = (0.0f32)
--      for j < d do
--        let z = query[j] - leaf_refs[i,j]
--        in dst + z*z
--    in  if dist <= radius
--        then res + query_w * leaf_ws[i]
--        else res

--  loop (res_w) = (1.0f32)
--    for (ref,w) in (zip leaf_refs ws) do
--      let dist = f32.sqrt <| sumSqrsSeq query ref
--      in  res_w * (if dist < radius then w else 1.0)

  map2(\ref i ->
        let dist = sumSqrsSeq query ref
        in  if dist <= radius then query_w * leaf_ws[i] else 0.0f32
      ) leaf_refs (iota m)
  |> reduce (+) 0.0f32


def sortQueriesByLeavesRadix [n] (num_bits: i32) (leaves: [n]i32) : ([n]i32, [n]i32) =
  -- (leaves, map i32.i64 (iota n))
  unzip <| radix_sort_by_key (\(l,_) -> l) num_bits i32.get_bit (zip leaves (map i32.i64 (iota n)))

def iterationSorted [q][n][d][num_leaves][ppl]
            (radius: f32)
            (h: i32)
            (kd_tree: [q](i32,f32,i32))
            (leaves:  [num_leaves][ppl][d]f32)
            (ws:      [num_leaves][ppl]f32)
            -- ^ invariant
            (queries: [n][d]f32)
            (query_ws:[n]f32)
            -- the loop state:
            (qleaves:     [n]i32)
            (stacks:      [n]i32)
            (dists:       [n]f32)
            (query_inds:  [n]i32)
            (res:  f32)
            -- ^ one weight per querry
          : ([n]i32, [n]i32, [n]f32, [n]i32, f32) =

  let queries_sorted = gather queries  query_inds
  let query_ws_sorted= gather query_ws query_inds

  -- apply brute force
  let new_res =
    map3 (\ query query_w leaf_ind ->
            if leaf_ind >= i32.i64 num_leaves
            then 0.0f32
            else bruteForce radius query query_w (leaves[leaf_ind]) (ws[leaf_ind])
              -- bruteForce1 radius query query_w res_w leaf_ind leaves ws
         ) queries_sorted query_ws_sorted qleaves
    |> reduce (+) 0.0f32 |> opaque

  -- start at old leaf and find a new leaf, until done!
  let (new_leaves, new_stacks, new_dists) = unzip3 <|
    map4 (\ query leaf_ind stack dist ->
            if leaf_ind >= i32.i64 num_leaves
            then
                 (leaf_ind, stack, dist)
            else traverseOnce radius h kd_tree query
                              (leaf_ind, stack, dist)
         ) queries_sorted qleaves stacks dists
    |> opaque

  let (qleaves', sort_inds) = sortQueriesByLeavesRadix (h+2) new_leaves
  -- we need (h+2) bits because the finish leaf is represented by num_leaves

  --let num_valid = map (\l -> if l < i32.i64 num_leaves then 1 else 0) new_leaves_all
  --                |> reduce_comm (+) 0i32 |> i64.i32

  let stacks'  = gather new_stacks sort_inds
  let dists'   = gather new_dists  sort_inds
  let query_inds' = gather query_inds sort_inds

  in  (qleaves', stacks', dists', query_inds', res + new_res)

def runIterRevAD [q][n][d][num_leaves][ppl]
            (radius: f32)
            (h: i32)
            (kd_tree: [q](i32,f32,i32))
            (leaves:  [num_leaves][ppl][d]f32)
            (ref_ws:  [num_leaves][ppl]f32)
            -- ^ invariant
            (queries: [n][d]f32)
            (query_ws:[n]f32)
            -- the loop state:
            (qleaves:     [n]i32)
            (stacks:      [n]i32)
            (dists:       [n]f32)
            (query_inds:  [n]i32)
            (res:  f32)
            -- ^ one weight per querry
          : ([num_leaves][ppl]f32, [n]f32) =
  let f (refe_ws , test_ws) : f32 =
    let (_,_,_,_, res) =
      iterationSorted radius h kd_tree leaves refe_ws queries test_ws qleaves stacks dists query_inds res
    in  res
  in  vjp f (ref_ws, query_ws) 1.0f32
