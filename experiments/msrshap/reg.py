import numpy as np
import scipy
from scipy.special import comb
import scipy.special

import sklearn.ensemble
import sklearn.linear_model

from .treeprob import tree_prob
from .base_estimator import BaseEstimator
from .p_generator import get_p
from .est_utils import combination_generator

# Combined regression class that supports:
# Leverage score sampling
# Paired sampling and non-paired sampling
# Sampling without replacement and sampling with replacement
# Constrained regression problem and unconstrained regression problem

def compute_weight_norm(w, p):
    # Faster implementation with vectorized operations
    n = len(p)
    binom_coeffs = comb(n - 1, np.arange(n))  # Vector of binomial coefficients
    w_inv_sq = 1 / w[:n]**2 + 1 / w[1:n+1]**2  # Vectorized inverse squares
    c_sq = np.sum(binom_coeffs * p**2 * w_inv_sq)
    return np.sqrt(c_sq)

def compute_weights(w1, p):
    n = len(p)
    w = np.zeros(n+1)
    w[0] = 1
    w[1] = w1
    # Solve for the rest of the weights
    for ell in range(1, n):
        numerator = p[ell]**2 * w[ell]**2 * w[ell-1]**2
        denominator1 = 2 * p[ell] * p[ell-1] * w[ell-1]**2
        denominator2 = -1*p[ell-1]**2 * w[ell]**2
        w_sq = numerator / (denominator1 + denominator2)
        if w_sq < 0: return np.full(n+1, np.nan)
        w[ell+1] = np.sqrt(w_sq)

    c = compute_weight_norm(w, p)
    # Check if c is nan
    #if not np.isnan(c):
    #    assert np.allclose(compute_weight_norm(w*c, p), 1)
    return w*c

def find_w1(p, start=10e-10, stop=10e10, num=1000):
    valid_w1s = []
    objectives = []
    for w1 in np.geomspace(start, stop, num):
        w = compute_weights(w1, p)
        if np.isnan(w).any(): continue
        objectives.append(
            ((w[::-1] - w)**2).sum()
        )
        valid_w1s.append(w1)
    closest_idx = np.argmin(np.array(objectives))
    return valid_w1s[closest_idx]

class UniversalRegression(BaseEstimator):
    def __init__(
        self,
        model,
        baseline,
        weighting,
        with_replace=False,
        constrain_reg=True,
    ):
        """
        Additional Parameters
        ----------
        pair_sampling : bool
            Whether to sample sets in pairs (S and its complement).
        with_replace: bool
            Whether to sample with replacement.
        constrain_reg:
            Whether to solve the constrained regression problem.
        """
        super().__init__(model, baseline, weighting)
        self.n = self.baseline.shape[1] # Number of features
        self.p = get_p(self.n, weighting)
        self.model = model
        self.baseline = baseline
        self.gen = np.random.Generator(np.random.PCG64())

        # Check if self.p is symmetric
        self.pair_sampling = np.allclose(self.p, self.p[::-1])

        self.with_replace = with_replace
        self.constrain_reg = constrain_reg
    
    def explain(self, explicand, num_samples, true_sum=None):
        self.explicand = explicand
        self.true_sum = true_sum

        self.select_samples = self.sample_with_replacement if self.with_replace else self.sample_without_replacement

        if self.constrain_reg:
            self.num_samples = (num_samples -2 ) // 2 * 2 # Two used for v1,v0
            self.valid_sizes = np.array(list(range(1, self.n)))
            self.leverage_score = lambda s : (self.p[s] + self.p[s-1]) * s * (self.n - s)
#            self.leverage_score = lambda s : np.ones_like(s)

            self.select_samples()
            self.compute_constrained()
        else:            
            self.num_samples = num_samples//2 * 2
            self.load_unconstrained()
            ## Add dimension to p for faster operations
            #self.p = np.append(self.p, -1e10)

            self.select_samples()
            self.compute_unconstrained()

        return self.phi
    
    def load_unconstrained(self):
        self.valid_sizes = np.array(list(range(self.n+1)))

        w1 = find_w1(self.p)
        self.w = compute_weights(w1, self.p)
#        print(f"w1: {w1}")
#        print(f"w: {self.w}")

        def leverage_score(s):
            p_ext = np.append(self.p, 0)
            lev = 0
            lev += (s > 0) * s * p_ext[s-1]**2
            lev += (s < self.n) * (self.n - s) * p_ext[s]**2
            lev /= self.w[s] ** 2
            return lev
        
#        leverage_score = lambda s : np.ones_like(s)

#        sum1, sum2 = 0, 0
#        for ell in range(self.n):
#            sum1 += scipy.special.binom(self.n-1, ell) * self.p[ell]**2
#            if ell >= 1:
#                sum2 += scipy.special.binom(self.n-2, ell-1) * (self.p[ell]-self.p[ell-1])**2
#        self.an = 2*sum1 - sum2
#        self.bn = sum2 / self.an
#    
#        def leverage_score(s):
#            first_term = 0 # |S| p_{|S|-1}^2 + (n-|S|) p_{|S|}^2
#            second_term = 0 # |S| p_{|S|-1} - (n-|S|) p_{|S|}
#            first_term += (s > 0) * s * self.p[s-1]**2
#            second_term += (s > 0) * s * self.p[s-1]
#            first_term += (s < self.n) * (self.n - s) * self.p[s]**2
#            second_term -= (s < self.n) * (self.n - s) * self.p[s]
#            #if s > 0:
#            #    first_term += s * self.p[s-1]**2
#            #    second_term += s * self.p[s-1]
#            #if s < self.n:
#            #    first_term += (self.n - s) * self.p[s]**2
#            #    second_term -= (self.n - s) * self.p[s]
#            # (first_term - bn / (1+n * bn) * second_term^2) / a_n
#            return (first_term - self.bn / (1+self.n * self.bn) * second_term**2) / self.an

        self.leverage_score = lambda s : np.array([leverage_score(val) for val in s])

    def update_prob(self):
        if self.with_replace:
            self.prob = self.leverage_score(self.valid_sizes)
            self.prob /= np.sum(self.prob)
        else: # Sampling without replacement
            lev = self.leverage_score(self.valid_sizes)
            if self.pair_sampling:
                lev += self.leverage_score(self.n - self.valid_sizes)
            self.prob = np.minimum(self.C * lev, np.ones_like(lev))
        
    def get_probs(self, s):
        if self.constrain_reg: # Length 1 is first prob
            return self.prob[s-1]
        return self.prob[s]
    
    def add_one_sample(self, idx, indices):
        if not self.pair_sampling:
            self.sampled[idx, indices] = 1
        else:            
            self.sampled[2*idx, indices] = 1
            indices_complement = np.array([i for i in range(self.n) if i not in indices])
            self.sampled[2*idx+1, indices_complement] = 1
    
    def sample_with_replacement(self):
        self.sampled = np.zeros((self.num_samples, self.n))
        num_sizes = self.num_samples if not self.pair_sampling else self.num_samples // 2
        self.update_prob() # Important to update self.prob
        sampled_sizes = self.gen.choice(self.valid_sizes, num_sizes, p=self.prob)
        for idx, s in enumerate(sampled_sizes):
            indices = self.gen.choice(self.n, s, replace=False)
            self.add_one_sample(idx, indices)
             
    def sample_without_replacement(self):
        self.find_constant_for_bernoulli()
        m_s_lookup = {}
        for s in self.valid_sizes:
            prob = self.get_probs(s)
            # Sample from Binomial distribution with (n choose s) trials and probability min(1, 2*C*sample_weight(s) / (n choose s))
            try:
                m_s = self.gen.binomial(int(scipy.special.binom(self.n, s)), prob)
            except OverflowError: # If the number of samples is too large, assume the number of samples is the expected number
                m_s = int(prob * scipy.special.binom(self.n, s))
            # DETERMINISTIC SAMPLING 
            m_s = int(prob * scipy.special.binom(self.n, s))
            if self.pair_sampling:
                if s == self.n // 2: # Already sampled all larger sets with the complement                    
                    if self.n % 2 == 0: # Special handling for middle set size if n is even
                        m_s_lookup[s] = m_s // 2
                    else: m_s_lookup[s] = m_s
                    break
            m_s_lookup[s] = m_s
        sampled_m = np.sum([m_s for m_s in m_s_lookup.values()])
        num_rows = sampled_m if not self.pair_sampling else sampled_m * 2
        self.sampled = np.zeros((num_rows, self.n))
        idx = 0
        for s, m_s in m_s_lookup.items(): 
            if self.pair_sampling and s == self.n // 2 and self.n % 2 == 0:
                # Partition the all middle sets into two
                # based on whether the combination contains n-1
                combo_gen = combination_generator(self.gen, self.n - 1, s-1, m_s)
                for indices in combo_gen:
                    self.add_one_sample(idx, list(indices) + [self.n-1])
                    idx += 1
            else:
                combo_gen = combination_generator(self.gen, self.n, s, m_s)
                for indices in combo_gen:
                    self.add_one_sample(idx, list(indices))
                    idx += 1

    def find_constant_for_bernoulli(self):
        # Choose C so that sampling without replacement from min(1, C*prob) gives the same expected number of samples
        self.C = 1/2**self.n # Assume at least n - 1 samples
        self.update_prob() # self.C used to update self.prob
        max_samples = 2**self.n
        if self.constrain_reg:
            max_samples -= 2 # Two for v0, v1
        m = min(self.num_samples, max_samples) # Maximum number of samples is 2^n -2
        # Find C with binary search
        L = 1 / 100**self.n
        R = 100 ** self.n
        binoms = scipy.special.binom(self.n, self.valid_sizes)
        expected = (binoms@self.prob).sum()
        while round(expected) != m:
            if expected < m: L = self.C
            else: R = self.C
            self.C = (L + R) / 2
            self.update_prob() # self.C used to update self.prob
            expected = (binoms@self.prob).sum()
    
    def compute_unconstrained(self):
        # x* = (A^T A)^-1 A^T v
        # xt = (A^T S^T W S A)^-1 A^T S^T W S v
        # phit = (1+bn 1 1^T) xt
        sizes = np.sum(self.sampled, axis=1).astype(int)
        self.sizes = sizes
        p_ext = np.append(self.p, 0)
        
        # Build A_tilde so A_tilde[S,i] = p[|S|-1]/w[|S|] if i in S else -p[|S|]/w[|S|]
        A_tilde = p_ext[sizes-1][:,None] * self.sampled - p_ext[sizes][:,None] * (1-self.sampled)
        A_tilde /= self.w[sizes][:,None]

        inputs = self.baseline * (1 - self.sampled) + self.explicand * self.sampled
        self.y = self.model.predict(inputs)
        Sv = self.y * self.w[sizes]
        self.prob_sampled = self.get_probs(sizes)

        reg_weighting = 1/self.prob_sampled

        ASSv = A_tilde.T @ np.diag(reg_weighting) @ Sv
        ASSA = A_tilde.T @ np.diag(reg_weighting) @ A_tilde
        self.phi = np.linalg.lstsq(ASSA, ASSv, rcond=None)[0]

    def compute_constrained(self):
        # A = Z P
        # b = v(z) - v0 - Z1 * sum(phi) / n
        # (A^T S^T S A)^-1 A^T S^T S b + (v1 - v0) / n
        # (P^T Z^T S^T S Z P)^-1 P^T Z^T S^T S b + (v1 - v0) / n

        self.sampled = self.sampled[np.sum(self.sampled, axis=1) != 0] # Remove zero rows
        sampled = self.sampled
        v0, v1 = self.model.predict(self.baseline), self.model.predict(self.explicand)
        self.v0, self.v1 = v0, v1
        inputs = self.baseline * (1 - sampled) + self.explicand * sampled
        self.y = self.model.predict(inputs)
        Sv = self.y - v0

        # Estimate constant term = 1/n sum_i phi_i
        # = 1/n sum_S (v(S)-v(emptyset)) (-p[|S|]*(n-|S|) + p[|S|-1] * |S|)
        # Compute probability each set size was sampled
        self.sizes = np.sum(sampled, axis=1).astype(int)
        sizes = self.sizes
        self.prob_sampled = self.get_probs(sizes)
        prob_sampled = self.prob_sampled


        sum_weighting = -self.p[sizes] * (self.n - sizes) + self.p[sizes-1] * sizes
        sum_phi = prob_sampled * sum_weighting @ Sv
        sum_phi = sum_phi + (v1-v0)*self.p[-1]*self.n

        if self.true_sum is not None:
            sum_phi = self.true_sum
        
        Sb = Sv - sizes * sum_phi / self.n

        # Projection matrix
        P = np.eye(self.n) - 1/self.n * np.ones((self.n, self.n))

        reg_weighting = 1/prob_sampled * (self.p[sizes-1] + self.p[sizes])

        PZSSb = P @ sampled.T @ np.diag(reg_weighting) @ Sb
        PZSSZP = P @ sampled.T @ np.diag(reg_weighting) @ sampled @ P
        PZSSZP_inv_PZSSb = np.linalg.lstsq(PZSSZP, PZSSb, rcond=None)[0]

        self.phi = PZSSZP_inv_PZSSb 
        self.phi += sum_phi / self.n

class UniversalConstrained(UniversalRegression):
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting, with_replace=False, constrain_reg=True)

class UniversalUnconstrained(UniversalRegression):
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting, with_replace=False, constrain_reg=False)
