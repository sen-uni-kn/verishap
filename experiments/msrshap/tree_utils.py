import xgboost
import numpy as np
import json
from .tree import Tree
import xgboost as xgb

def is_sklearn_random_forest(model):
    return (
        "sklearn.ensemble._forest.RandomForestRegressor" in str(type(model))
        or "sklearn.ensemble._forest.RandomForestClassifier" in str(type(model))
    )

def is_xgboost_model(model):
    return xgboost is not None and (
        isinstance(model, xgboost.XGBRegressor) or isinstance(model, xgboost.XGBClassifier)
    )

def _collect_nodes(node, mapping):
    """
    Recursively DFS through the JSON tree and flatten into a dict
      { nodeid: node_dict }
    preserving the exact visit order.
    """
    mapping[node["nodeid"]] = node
    for child in node.get("children", []):
        _collect_nodes(child, mapping)

def _build_single_tree(tree_json, n_features, name_to_index):
    """
    Build one Tree from either:
      - a JSON string (from Booster.get_dump)
      - an already-parsed dict (for unit tests)
    """
    if isinstance(tree_json, str):
        root = json.loads(tree_json)
    else:
        root = tree_json

    # Flatten via DFS
    nodes = {}
    _collect_nodes(root, nodes)

    # Map each raw nodeid → row index in our numpy arrays
    #   (in the same DFS order we saw them)
    id_to_idx = {nid: i for i, nid in enumerate(nodes)}
    n_nodes   = len(nodes)

    thresholds = np.zeros(n_nodes, dtype=np.float32)
    values     = np.zeros(n_nodes, dtype=np.float32)
    features   = np.full (n_nodes, -1,       dtype=np.int32)
    children_left  = np.full(n_nodes, -1, dtype=np.int32)
    children_right = np.full(n_nodes, -1, dtype=np.int32)

    for nid, node in nodes.items():
        idx = id_to_idx[nid]

        if "leaf" in node:
            # leaf node: store its output value
            values[idx] = np.float32(node["leaf"])
        else:
            # internal split node
            split_key = node["split"]
            if split_key.startswith("f") and split_key[1:].isdigit():
                feat_idx = int(split_key[1:])
            else:
                feat_idx = name_to_index[split_key]

            features[idx]   = feat_idx
            thresholds[idx] = np.float32(node.get("split_condition", 0.5))
            children_left[idx]  = id_to_idx[node["yes"]]  # YES branch = ≤
            children_right[idx] = id_to_idx[node["no"]]   # NO branch = >

    return Tree(children_left,
                children_right,
                features,
                thresholds,
                values,
                n_features)

def parse_xgb_tree(model) -> list[Tree]:
    booster = model if isinstance(model, xgb.Booster) else model.get_booster()
    feature_names = booster.feature_names
    name_to_index = {name: i for i, name in enumerate(feature_names)}

    json_dumps = booster.get_dump(dump_format="json")
    n_features = booster.num_features()

    return [
        _build_single_tree(js, n_features, name_to_index)
        for js in json_dumps
    ]