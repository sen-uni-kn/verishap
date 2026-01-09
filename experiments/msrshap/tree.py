#  Internal Tree Representation

class Tree:
    """
    A simple tree representation for the interventional recursion:
    - children_left[node], children_right[node]: index of left/right child or -1 if leaf
    - features[node], thresholds[node]: which feature + threshold used at node
    - values[node]: the leaf value (if leaf), 0 if internal node
    - n_features: total number of features in the data
    """
    def __init__(self, children_left, children_right, features, thresholds, values, n_features):
        self.children_left = children_left
        self.children_right = children_right
        self.features = features
        self.thresholds = thresholds
        self.values = values
        self.n_features = n_features

    def is_leaf(self, node):
        return (self.children_left[node] == -1 and 
                self.children_right[node] == -1)