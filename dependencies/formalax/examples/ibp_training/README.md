# IBP Training
The notebooks in this folder implement IBP training ([Gowal et al., 2019](https://arxiv.org/pdf/1810.12715)).
In IBP training, IBP is used to compute a upper bound on the adversarial loss and the model is trained to minimize this upper bound.
The notebooks in this folder implement the basic technique from [Gowal et al., 2019](https://arxiv.org/pdf/1810.12715)
and the IBP initialization and architecture improvements from [Shi et al., 2021](https://proceedings.neurips.cc/paper/2021/hash/988f9153ac4fd966ea302dd9ab9bae15-Abstract.html).

## ibp_mnist Notebook
This notebook implements IBP training on the MNIST and similar datasets.
The following results were obtained using this notebook with the default hyperparameters
in the notebook or the `.yaml` hyperparameter files for the corresponding dataset 
on Ubuntu 24.04 using an NVIDIA GeForce RTX™ 3070 Laptop GPU with 8 GiB of memory.

| Dataset      | Eps | Natural Accuracy | Certified Accuracy | Runtime | Hyperparameters      |
|--------------|-----|------------------|--------------------|---------|----------------------|
| MNIST        | 0.1 | 96.45%           | 89.40%             | ~10min  | Default              |
| EMNIST       | 0.1 | 74.02%           | 63.07%             | ~20min  | `emnist_params.yaml` |
| FashionMNIST | 0.1 | 76.14%           | 64.79%             | ~5min   | Default              |

In this table, `Eps` is the perturbation radius, both the natural (clean/standard) accuracy and the certified accuracy are on the test set.
The certified accuracy is computed using IBP.
