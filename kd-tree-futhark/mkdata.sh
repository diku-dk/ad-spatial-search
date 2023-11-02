futhark opencl mkdata.fut
cat ./data/sqrad-dot01-leaf-256-refs-512Kx3-dot1-10dot1-ws-dot1-dot2-queries-1M.in | ./mkdata --entry=brute_primal > data/5radiuses-brute-force-primal-refs-512K-queries-1M.out
cat ./data/sqrad-dot01-leaf-256-refs-512Kx3-dot1-10dot1-ws-dot1-dot2-queries-1M.in | ./mkdata --entry=brute_revad > data/5radiuses-brute-force-revad-refs-512K-queries-1M.out
cat ./data/sqrad-dot01-leaf-256-refs-512Kx3-dot1-10dot1-ws-dot1-dot2-queries-1M.in | ./mkdata --entry=bruteForce_input > data/5radiuses-brute-force-input-refs-512K-queries-1M.out
cat ./data/kdtree-prop-refs-512K-queries-1M.in | ./mkdata --entry=iterationSorted_input > data/5radiuses-iterationSorted-refs-512K-queries-1M.in
