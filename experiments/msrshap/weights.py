import numpy as np
from scipy.special import beta as beta_function
import numba

def calc_weight(U, V, weighting="banzhaf"):
    """
    Calculate the weight of Shapley, Banzhaf, or Beta Shapley value.

    Parameters
    ----------
    U : int
        # of features that matched the foreground sample along the path
    V : int
        total # of unique features encountered along the path
    weighting : str
        The type of weight to calculate: 'shapley', 'banzhaf', 'beta_shapley_[alpha]_[beta]', 'weighted_banzhaf_[w]'

    Returns 
    -------
    float
        The weight of the specified measure.
    """
    weighting_list = weighting.split('_')
    if "beta" in weighting:
        # handle 'beta_shapley', 'beta_shapley_4_1', etc.
        if len(weighting_list) == 2:
            # e.g. 'beta_shapley' alone => default alpha=1, beta=1
            alpha, beta_ = (1, 1)
        else:
            alpha_str, beta_str = weighting_list[-2], weighting_list[-1]
            alpha, beta_ = (float(alpha_str), float(beta_str))
        return w_beta_shapley(V, U, alpha, beta_)

    if "weighted" in weighting:
        # e.g. 'weighted_banzhaf' => default=0.5
        # or 'weighted_banzhaf_0.3'
        if len(weighting_list) == 2:
            w = 0.5
        else:
            w = float(weighting_list[-1])
        return w_weighted_banzhaf(V, U, w)

    if weighting == "shapley":
        return w_shapley(V, U)
    elif weighting == "banzhaf":
        return w_banzhaf(V)
    else:
        raise ValueError(f"Invalid weighting type: {weighting}")
    

@numba.jit(nopython=True)
def factorial(n):
    """
    Helper: A simple factorial for n >= 0, used for calc_weight.
    """
    if n < 2:
        return 1.0
    result = 1.0
    for i in range(2, n + 1):
        result *= i
    return result


# def beta_function(a, b):
#     """Helper: Calculate the Beta function."""
#     return gamma(a) * gamma(b) / gamma(a + b)


def w_shapley(V, U):
    """Calculate the weight of Shapley value."""
    num = factorial(U) * factorial(V - U - 1)
    den = factorial(V)
    return num / den


def w_beta_shapley(V, U, alpha, beta):
    """Calculate the weight of beta Shapley value."""
    j = U + 1  # |S| = j - 1
    num = beta_function(j + beta - 1, V - j + alpha)
    den = beta_function(alpha, beta)
    return num / den


def w_banzhaf(V):
    """Calculate the weight of Banzhaf value."""
    return 1 / (2 ** (V - 1))


def w_weighted_banzhaf(V, U, weight):
    """Calculate the weight of weighted Banzhaf value."""
    return (weight ** U) * ((1 - weight) ** (V - U - 1))