from solve_kpdfs_instance_dp import compute_A
import numpy as np
item_profits = np.array([3,5,6,8])
item_weights = np.array([1,3,4,1])
capacity = 7
set_size = len(item_profits)
compute_A(item_profits,item_weights,capacity,set_size,test=True)

"""
for t, (p, w) in enumerate(zip(item_profits, item_weights)):
    if w > capacity:
        continue
    A[w:, 1:] = np.maximum(A[w:, 1:], A[:capacity + 1 - w, :-1] + p)
"""

