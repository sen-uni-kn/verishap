import math
import numpy as np
from typing import Tuple, Dict, Literal
import re

def ith_combination(pool, r, index):
    # Function written by ChatGPT
    """
    Compute the index-th combination (0-based) in lexicographic order
    without generating all previous combinations.
    """
    n = len(pool)
    combination = []
    elements_left = n
    k = r
    start = 0
    
    for i in range(r):
        # Find the largest value for the first element in the combination
        # that allows completing the remaining k-1 elements
        for j in range(start, elements_left):
            count = math.comb(elements_left - j - 1, k - 1)
            if index < count:
                combination.append(pool[j])
                k -= 1
                start = j + 1
                break
            index -= count
    
    return tuple(combination)

def combination_generator(gen, n, s, num_samples):
    """
    Generate num_samples random combinations of s elements from a pool num_samples of size n in two settings:
    1. If the number of combinations is small (converting to an int does NOT cause an overflow error), randomly sample num_samples integers without replacement and generate the corresponding combinations on the fly with ith_combination.
    2. If the number of combinations is large (converting to an int DOES cause an overflow error), randomly sample num_samples combinations directly with replacement.
    """
    num_combos = math.comb(n, s)
    try:
        indices = gen.choice(num_combos, num_samples, replace=False)
        for i in indices:
            yield ith_combination(range(n), s, i)
    except OverflowError:
        for _ in range(num_samples):
            yield gen.choice(n, s, replace=False)