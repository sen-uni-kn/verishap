import numpy as np

from .tree import Tree
from .tree_utils import is_sklearn_random_forest, is_xgboost_model, parse_xgb_tree
from .weights import calc_weight
import warnings
import xgboost
import json

def tree_prob(baseline, explicands, model, weighting="shapley"):
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar multiply")
    true_values = []
    explainer = TreeProbExplainer(model=model, baseline=baseline, weighting=weighting)
    for explicand in explicands:
        true_values.append(explainer.explain(explicand))
    return true_values


def prob_recurse(
    node_index,
    U,
    V,
    xlist,
    clist,
    x,
    c,
    children_left,
    children_right,
    features,
    thresholds,
    values,
    phi,
    model_type,
    weighting="shapley"
):
    """
    The two-path recursion from Algorithm 3 in:
      https://www.nature.com/articles/s42256-019-0138-9
    
    This traverses a single decision tree, 
    handling forced path splits when x and c differ.
    """
    # Leaf node => compute partial contribution
    if children_left[node_index] == -1 and children_right[node_index] == -1:
        vj = values[node_index]
        if V == 0:
            return (0.0, 0.0)
        else:
            pos_weight = calc_weight(U-1, V, weighting)
            pos = pos_weight * vj
            neg_weight = calc_weight(U, V, weighting)
            neg = -neg_weight * vj if not np.isnan(neg_weight) else 0.0
            return (pos, neg)

    # Internal node
    dj = features[node_index]  # feature index for this split
    tj = thresholds[node_index]
    a_j = children_left[node_index]
    b_j = children_right[node_index]

    xdj = x[dj]
    cdj = c[dj]

    # If xlist[dj] > 0 => forcibly follow x's path
    if xlist[dj] > 0:
        split = xdj > tj if model_type == "rf" else xdj >= tj
        if split:
            return prob_recurse(
                b_j, U, V, xlist, clist, x, c, 
                children_left, children_right,
                features, thresholds, values,
                phi, model_type, weighting
            )
        else:
            return prob_recurse(
                a_j, U, V, xlist, clist, x, c,
                children_left, children_right,
                features, thresholds, values,
                phi, model_type, weighting
            )

    # If clist[dj] > 0 => forcibly follow c's path
    if clist[dj] > 0:
        split = cdj > tj if model_type == "rf" else cdj >= tj
        if split:
            return prob_recurse(
                b_j, U, V, xlist, clist, x, c,
                children_left, children_right,
                features, thresholds, values,
                phi, model_type, weighting
            )
        else:
            return prob_recurse(
                a_j, U, V, xlist, clist, x, c,
                children_left, children_right,
                features, thresholds, values,
                phi, model_type, weighting
            )
    
    # If x and c both go the same way => single path
    if model_type == "rf":
        both_right = (xdj > tj) and (cdj > tj)
        both_left = (xdj <= tj) and (cdj <= tj)
    elif model_type == "xgb":
        both_right = (xdj >= tj) and (cdj >= tj)
        both_left = (xdj < tj) and (cdj < tj)

    if both_right:
        return prob_recurse(
            b_j, U, V, xlist, clist, x, c,
            children_left, children_right,
            features, thresholds, values,
            phi, model_type, weighting
        )

    if both_left:
        return prob_recurse(
            a_j, U, V, xlist, clist, x, c,
            children_left, children_right,
            features, thresholds, values,
            phi, model_type, weighting
        )

    # x and c differ => "split path"
    # 1) x -> right, c -> left
    if model_type == "rf":
        split = (xdj > tj) and (cdj <= tj)
    elif model_type == "xgb":
        split = (xdj >= tj) and (cdj < tj)
    if split:
        # Force x down right child
        xlist[dj] += 1
        posx, negx = prob_recurse(
            b_j, U+1, V+1, xlist, clist, x, c,
            children_left, children_right,
            features, thresholds, values,
            phi, model_type, weighting
        )
        xlist[dj] -= 1

        # Force c down left child
        clist[dj] += 1
        posc, negc = prob_recurse(
            a_j, U, V+1, xlist, clist, x, c,
            children_left, children_right,
            features, thresholds, values,
            phi, model_type, weighting
        )
        clist[dj] -= 1

        phi[dj] += (posx + negc)
        return (posx + posc, negx + negc)

    # 2) x -> left, c -> right
    if model_type == "rf":
        split = (xdj < tj) and (cdj >= tj)
    elif model_type == "xgb":
        split = (xdj < tj) and (cdj >= tj)
    if split:
        xlist[dj] += 1
        posx, negx = prob_recurse(
            a_j, U+1, V+1, xlist, clist, x, c,
            children_left, children_right,
            features, thresholds, values,
            phi, model_type, weighting
        )
        xlist[dj] -= 1

        clist[dj] += 1
        posc, negc = prob_recurse(
            b_j, U, V+1, xlist, clist, x, c,
            children_left, children_right,
            features, thresholds, values,
            phi, model_type, weighting
        )
        clist[dj] -= 1
        
        phi[dj] += (posx + negc)
        return (posx + posc, negx + negc)

    # If we get here, something unexpected occurred
    return (0.0, 0.0)


class TreeProbExplainer:
    """
    Interventional Tree Probabilistic Value Explainer
    using the two-path recursion (Algorithm 3)
    for either:
      - RandomForestRegressor from sklearn
      - XGBRegressor (XGBoost)
    """

    def __init__(self, model, baseline, weighting="shapley", **kwargs):
        self.baseline = baseline
        self.weighting = weighting
        self.trees = []

        self._is_rf  = is_sklearn_random_forest(model)
        self._is_xgb = is_xgboost_model(model)

        # -- Detect model type
        if self._is_rf:
            # scikit-learn RandomForest
            n_features = model.n_features_in_
            for estimator in model.estimators_:
                t_ = estimator.tree_
                tree_obj = Tree(
                    children_left=t_.children_left,
                    children_right=t_.children_right,
                    features=t_.feature,
                    thresholds=t_.threshold,
                    values=t_.value[:, 0, 0],   # shape (n_nodes, 1, 1) => flatten
                    n_features=n_features
                )
                self.trees.append(tree_obj)

        elif self._is_xgb:
            self.trees = parse_xgb_tree(model)
        else:
            raise ValueError(
                "This explainer only supports scikit-learn RandomForestRegressor or XGBoost (XGBRegressor/XGBClassifier)."
            )

        weighting_list = weighting.split('_')
        if "beta" in weighting:
            if len(weighting_list) == 2:
                print("No (alpha,beta) specified, defaulting to (1,1). You can do e.g. 'beta_shapley_4_1'.")
            # else:
                # print(f"Using Beta Shapley with alpha={weighting_list[-2]}, beta={weighting_list[-1]}.")
        elif "weighted" in weighting:
            if len(weighting_list) == 2:
                print("No weight specified, defaulting to 0.5. You can do e.g. 'weighted_banzhaf_0.3'.")
            # else:
                # print(f"Using Weighted Banzhaf with weight={weighting_list[-1]}.")
    

    def explain(self, explicand, **kwargs):
        """
        Calculate interventional prob values for each row in X,
        averaged over all reference vectors in baseline

        Parameters
        ----------
        X : numpy.ndarray (n_samples, n_features)
            The data for which we want SHAP-like values.
        baseline : np.ndarray (n_baseline, n_features)
            Baseline/reference samples.

        Returns
        -------
        phi : np.ndarray of shape (n_samples, n_features)
            The final array of attributions. 
        """
        if "pandas" in str(type(explicand)):
            explicand = explicand.values
        if len(explicand.shape) == 1:
            explicand = explicand.reshape(1, -1)

        if self._is_rf:
            model_type = "rf"
        elif self._is_xgb:
            model_type = "xgb"
        else:
            raise ValueError(f"Invalid model type: {self._is_rf} or {self._is_xgb}")
        
        baseline = np.array(self.baseline)
        if len(baseline.shape) == 1:
            baseline = baseline.reshape(1, -1)

        n_samples, n_features = explicand.shape
        phi = np.zeros((n_samples, n_features), dtype=np.float64)

        for i in range(n_samples):
            # Accumulate across all trees
            row_phi = np.zeros(n_features, dtype=np.float64)
            for t in self.trees:
                sum_phi_over_c = np.zeros(n_features, dtype=np.float64)
                for c_idx in range(baseline.shape[0]):
                    xlist = np.zeros(n_features, dtype=np.int64)
                    clist = np.zeros(n_features, dtype=np.int64)

                    # Recurse from the root node = 0, with U=0, V=0
                    prob_recurse(
                        node_index=0, 
                        U=0, 
                        V=0,
                        xlist=xlist, 
                        clist=clist,
                        x=explicand[i, :],
                        c=baseline[c_idx, :],
                        children_left=t.children_left,
                        children_right=t.children_right,
                        features=t.features,
                        thresholds=t.thresholds,
                        values=t.values,
                        phi=sum_phi_over_c,
                        model_type=model_type,
                        weighting=self.weighting
                    )

                # Average over all baseline samples
                sum_phi_over_c /= baseline.shape[0]
                row_phi += sum_phi_over_c

            # Average over all trees
            if self._is_rf:     
                row_phi /= len(self.trees)

            phi[i, :] = row_phi

        return phi