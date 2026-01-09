import numpy as np
from math import factorial
from scipy.special import beta as beta_function
from scipy.special import comb as scipy_comb 

def get_p(n, weighting="shapley"):
    weighting_list = weighting.split('_')

    if "beta" in weighting:
        if len(weighting_list) == 2:
            alpha, beta = (1, 1)
        else:
            alpha_str, beta_str = weighting_list[-2], weighting_list[-1]
            alpha, beta = (float(alpha_str), float(beta_str))
        weighting = "beta_shapley"
        return beta_shapley_distribution(n, alpha, beta)
    
    elif "weighted" in weighting:
        if len(weighting_list) == 2:
            p_weight = 0.5
        else:
            p_weight = float(weighting_list[-1])
        weighting = "weighted_banzhaf"
        return weighted_banzhaf_distribution(n, p_weight)

    elif "random" in weighting:
        random_state = weighting_list[-1] if len(weighting_list) == 2 else 42
        weighting = "random"
        return random_probabilistic_distribution(n, int(random_state))
    
    if weighting == "shapley":
        return shapley_distribution(n)
    
    elif weighting == "banzhaf":
        return banzhaf_distribution(n)
    
    else:
        raise ValueError(f"Unknown weighting type: {weighting}")


# 1) Randomly generate p that satisfies sum_{k=0}^{n-1} binom(n-1,k)* p[k] = 1
def random_probabilistic_distribution(n, random_state=None):
    """
    Generate a random length-n array p = [p_0, ..., p_{n-1}]
    s.t. sum_{k=0}^{n-1} binom(n-1, k) * p[k] = 1 and each p[k] >= 0.

    Parameters
    ----------
    n : int
        Number of components in the distribution (p_0 through p_{n-1}).
    random_state : int or None
        If not None, sets the np.random.seed for reproducibility.

    Returns
    -------
    p : np.ndarray
        A length-n array of probabilities satisfying the probabilistic value constraints.
    """
    if random_state is not None:
        np.random.seed(random_state)

    x = np.random.rand(n)

    # Compute binomial coefficients for n-1 choose k, k=0..n-1
    binom_coeffs = scipy_comb(n - 1, np.arange(n))

    # Compute the normalizing denominator: sum(binom(n-1, k) * x[k] for k)
    denominator = np.dot(binom_coeffs, x)
    
    if denominator == 0:
        raise ValueError("Denominator in normalization is zero. Choose a different random seed or check n.")

    p = x / denominator

    return p


# 2) Distributions p for four special semivalues
def shapley_distribution(n):
    """
    Generate the distribution p_k for the Shapley value.
    The standard semivalue weight is: w(k,n) = k! * (n - k -1)! / n!
    Then p_k = w(k,n) / binom(n-1, k).

    Parameters
    ----------
    n : int
        Total number of players.

    Returns
    -------
    p : np.ndarray
        Shapley distribution of length n (p_0, p_1, ..., p_{n-1}).
    """
    k = np.arange(n)
    # w(k,n) = k! * (n - k -1)! / n!
    # Handle the case when n - k -1 < 0 by setting w_k to 0
    with np.errstate(over='ignore'):
        p_k = np.array([factorial(ki) * factorial(n - ki - 1) / factorial(n) if (n - ki -1) >=0 else 0.0 for ki in k])

    return p_k

def banzhaf_distribution(n):
    """
    Generate the distribution p_k for the Banzhaf value.
    The standard semivalue weight is: w(k,n) = 1 / 2^(n-1)
    Then p_k = w(k,n) / binom(n-1, k).

    Parameters
    ----------
    n : int
        Total number of players.

    Returns
    -------
    p : np.ndarray
        Banzhaf distribution of length n (p_0, p_1, ..., p_{n-1}).
    """
    return np.full(n, 1.0 / (2 ** (n - 1)))

def beta_shapley_distribution(n, alpha, beta):
    """
    Generate the distribution p_k for the Beta Shapley value.
    Weight: w(k,n) = Beta(k + beta, n - k -1 - alpha) / Beta(alpha, beta)

    Then p_k = w(k,n) / binom(n-1, k).

    Parameters
    ----------
    n : int
        Total number of players.
    alpha : float
    beta : float

    Returns
    -------
    p : np.ndarray
        Beta Shapley distribution p_k of length n.
    """
    k = np.arange(n)
    # w(k,n) = Beta(k + beta, (n -1 -k) + alpha) / Beta(alpha, beta)
    num = np.array([
        beta_function(ki + beta, (n - 1 - ki) + alpha)
        for ki in k
    ])
    den = beta_function(alpha, beta)

    with np.errstate(divide='ignore', invalid='ignore'):
        p_k = np.where(den != 0, num / den, 0.0)

    return p_k

def weighted_banzhaf_distribution(n, weight):
    """
    Generate the distribution p_k for a Weighted Banzhaf value with parameter p_weight.
    Weight: w(k,n) = p_weight^(k-1) * (1 - p_weight)^(n - k)

    Then p_k = w(k,n) / binom(n-1, k).

    Parameters
    ----------
    n : int
        Total number of players.
    p_weight : float
        "p" from the table, 0 <= p_weight <= 1

    Returns
    -------
    p : np.ndarray
        Weighted Banzhaf distribution p_k of length n.
    """
    if not (0 <= weight <= 1):
        raise ValueError("p_weight must be between 0 and 1.")

    k = np.arange(n)

    # w(k,n) = p_weight^(k-1) * (1 - p_weight)^(n - k)
    p_k = weight ** k * (1 - weight) ** (n - k - 1)

    return p_k


def check_sum_is_one(p, n, tol=1e-8):
    if len(p) != n:
        raise ValueError(f"Length of p ({len(p)}) does not match n ({n}).")

    binom_coeffs = scipy_comb(n - 1, np.arange(n))
    sum_value = np.dot(binom_coeffs, p)

    is_valid = np.abs(sum_value - 1.0) <= tol
    return is_valid, sum_value


# Test
if __name__ == "__main__":
    n = 5

    print("1. Random Probabilistic Distribution (n=5):")
    p_rand = random_probabilistic_distribution(n, random_state=42)
    print("p_rand =", p_rand)
    is_valid, sum_val = check_sum_is_one(p_rand, n)
    print(f"Sum of binom(n-1,k)*p_k = {sum_val:.10f} | Valid: {is_valid}\n")

    print("2. Shapley Distribution (n=5):")
    p_shap = shapley_distribution(n)
    print("Shapley p =", p_shap)
    is_valid, sum_val = check_sum_is_one(p_shap, n)
    print(f"Sum of binom(n-1,k)*p_k = {sum_val:.10f} | Valid: {is_valid}\n")

    print("3. Banzhaf Distribution (n=5):")
    p_bz = banzhaf_distribution(n)
    print("Banzhaf p =", p_bz)
    is_valid, sum_val = check_sum_is_one(p_bz, n)
    print(f"Sum of binom(n-1,k)*p_k = {sum_val:.10f} | Valid: {is_valid}\n")

    alpha_val = 1.0
    beta_val = 2.0
    print(f"4. Beta Shapley Distribution (n=5, alpha={alpha_val}, beta={beta_val}):")
    p_beta = beta_shapley_distribution(n, alpha_val, beta_val)
    print("Beta Shapley p =", p_beta)
    is_valid, sum_val = check_sum_is_one(p_beta, n)
    print(f"Sum of binom(n-1,k)*p_k = {sum_val:.10f} | Valid: {is_valid}\n")

    weight = 0.5
    print(f"5. Weighted Banzhaf Distribution (n=5, p={weight}):")
    p_wbz = weighted_banzhaf_distribution(n, weight)
    print("Weighted Banzhaf p =", p_wbz)
    is_valid, sum_val = check_sum_is_one(p_wbz, n)
    print(f"Sum of binom(n-1,k)*p_k = {sum_val:.10f} | Valid: {is_valid}\n")