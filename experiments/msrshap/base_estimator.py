import warnings
import numpy as np
from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    """
    Abstract base class for probabilistic value estimators.
    """
    def __init__(self, model, baseline, weighting):
        """
        Parameters
        ----------
        model : callable or sklearn-like object
            Must have `predict()` that takes shape (n_samples, n_features).
        baseline : np.ndarray
            The baseline point for feature perturbations.
        weighting : str
            The type of probabilistic value to estimate.
            Options: {"shapley", "banzhaf", "beta_shapley_[alpha]_[beta]", "weighted_banzhaf_[p_weight], random_[seed]"}
        """
        warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
        self.model = model
        self.baseline = baseline
        self.weighting = weighting

    @abstractmethod
    def explain(self, explicand, num_samples):
        """
        Compute the attributions/feature importances for
        the given baseline, explicand, and model.

        Parameters
        ----------
        explicand : np.ndarray
            The instance (or instances) to be explained.
        num_samples : int
            Number of samples to draw for estimator.

        Returns
        -------
        phi : np.ndarray
            The final attributions for each feature 
            (dimensions may vary across methods).
        """
        pass