#  Copyright (c) 2025. The Formalax Authors.
#  Licensed under the MIT license.
import shap


def exact_shap(model, baseline, x, silent=False):
    """The Exact SHAP explainer from the `shap` library.

    Computes exact SHAP values using enumeration.
    """
    explainer = shap.ExactExplainer(model, baseline)
    shap_values = explainer(x, max_evals=None, silent=silent)
    return shap_values


def kernel_shap(model, baseline, x, num_samples=1024, silent=False):
    """The Kernel SHAP explainer from the `shap` library."""
    explainer = shap.KernelExplainer(model, baseline)
    shap_values = explainer.shap_values(
        x, nsamples=num_samples, silent=silent, l1_reg=False
    )
    return shap_values


def permutation_shap(model, baseline, x, num_samples=1024, silent=False):
    """The Permutation SHAP explainer from the `shap` library."""
    x = x.numpy().astype("float64")
    num_features = x.shape[-1]
    num_permutations = num_samples // num_features

    explainer = shap.PermutationExplainer(model, baseline)
    shap_values = explainer.shap_values(
        x, npermutations=num_permutations, silent=silent
    )
    return shap_values


def sampling_shap(model, baseline, x, num_samples=1024, silent=False):
    """The Sampling SHAP explainer from the `shap` library."""
    explainer = shap.SamplingExplainer(model, baseline)
    shap_values = explainer.shap_values(x, nsamples=num_samples, silent=silent)
    return shap_values
