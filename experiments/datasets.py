# Copyright 2025 David Boetius
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import kagglehub
import numpy as np
import pandas as pd
import shap.datasets
import torch
from PIL import Image
from torch.utils.data import Dataset
from ucimlrepo import fetch_ucirepo


def load_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads a tabular dataset."""
    if dataset.startswith("corrgroups"):
        num_features = int(dataset[-2:])
        data, targets = corrgroups(num_features)
    elif dataset.startswith("independentlinear"):
        num_features = int(dataset[-2:])
        data, targets = independentlinear(num_features)
    elif hasattr(globals(), dataset):
        data, targets = getattr(globals(), dataset)()
    elif hasattr(shap.datasets, dataset):
        data, targets = getattr(shap.datasets, dataset)()
        data = data.to_numpy().astype(np.float32)
        data = np.nan_to_num(data)
    else:
        raise ValueError(f"Dataset {dataset} not found.")
    return data, targets


def corrgroups(
    num_features: int = 60, n_points: int = 1_000
) -> tuple[np.ndarray, np.ndarray]:
    """Correlated Groups Dataset from shap.

    This dataset is adapted from the `corrgroups60` dataset in the `shap` library.
    https://github.com/shap/shap/blob/83140b120d66e24377d5c8d59186f3a815cb4304/shap/datasets.py
    Original copyright by Scott Lundberg (2018).

    Synthetic datasets consisting of features with tight correlations
    among distinct groups of features.
    The number of features can be chosen arbitrarily.

    Parameters
    ----------
    num_features : int, optional
        Number of features to generate. Default is 60.
    n_points : int, optional
        Number of data points to generate. Default is 1,000.

    Returns
    -------
    x : np.ndarray
        The feature data matrix
    y : np.ndarray
        The target variables

    Notes
    -----
    - The dataset is generated with known correlations among distinct groups of features.
    - Each feature is a unit variance Gaussian random variable centred around 0.
    - The labels are generated based on a linear function of the features with added random noise.

    Examples
    --------
    .. code-block:: python

        data, target = corrgroups60()

    """
    # set a constant seed
    old_seed = np.random.seed()
    np.random.seed(0)

    # generate dataset with known correlation
    n, m = n_points, num_features

    # set one coefficient from each group of 3 to 1
    beta = np.zeros(m)
    beta[0 : n // 2 : 3] = 1

    # build a correlation matrix with groups of 3 tightly correlated features
    c = np.eye(m)
    for i in range(0, m // 2, 3):
        c[i, i + 1] = c[i + 1, i] = 0.99
        c[i, i + 2] = c[i + 2, i] = 0.99
        c[i + 1, i + 2] = c[i + 2, i + 1] = 0.99

    # Make sure the sample correlation is a perfect match
    x_start = np.random.randn(n, m)
    x_centered = x_start - x_start.mean(0)
    sigma = np.matmul(x_centered.T, x_centered) / x_centered.shape[0]
    w = np.linalg.cholesky(np.linalg.inv(sigma)).T
    x_white = np.matmul(x_centered, w.T)
    assert (
        np.linalg.norm(np.corrcoef(np.matmul(x_centered, w.T).T) - np.eye(m)) < 1e-6
    )  # ensure this decorrelates the data

    # create the final data
    x = np.matmul(x_white, np.linalg.cholesky(c).T)
    y = np.matmul(x, beta) + np.random.randn(n) * 1e-2

    # restore the previous numpy random seed
    np.random.seed(old_seed)

    return x, y


def independentlinear(
    num_features: int = 60, n_points: int = 1_000
) -> tuple[np.ndarray, np.ndarray]:
    """Independent Linear Dataset from shap.

    This dataset is adapted from the `independentlinear60` dataset in the `shap` library.
    https://github.com/shap/shap/blob/83140b120d66e24377d5c8d59186f3a815cb4304/shap/datasets.py
    Original copyright by Scott Lundberg (2018).

    A synthetic dataset with an arbitrary number of features.

    Parameters
    ----------
    num_features : int, optional
        Number of features to generate. Default is 60.
    n_points : int, optional
        Number of data points to generate. Default is 1,000.

    Returns
    -------
    x : np.ndarray
        The feature data matrix
    y : np.ndarray
        The target variables

    Notes
    -----
    - Each feature is a unit variance Gaussian random variable centred around 0.
    - The labels are generated based on a linear function of the features with added random noise.

    Examples
    --------
    .. code-block:: python

        features, labels = shap.datasets.independentlinear60()

    """
    # set a constant seed
    old_seed = np.random.seed()
    np.random.seed(0)

    # generate dataset with known correlation
    n, m = n_points, num_features

    # set one coefficient from each group of 3 to 1
    beta = np.zeros(m)
    beta[0 : n // 2 : 3] = 1

    # Make sure the sample correlation is a perfect match
    x_start = np.random.randn(n, m)
    x = x_start - x_start.mean(0)
    y = np.matmul(x, beta) + np.random.randn(n) * 1e-2

    # restore the previous numpy random seed
    np.random.seed(old_seed)

    return x, y


# ==============================================================================
# UCI ML Repository Datasets
# ==============================================================================


def categorical_to_int(x: pd.DataFrame) -> pd.DataFrame:
    for col in x.select_dtypes(include=["object"]).columns:
        x.loc[:, col] = x[col].astype("category").cat.codes
    return x


def save_dataset(dataset_name: str):
    def decorator(fn):
        def wrapper(*args, root=".datasets", **kwargs):
            file = Path(root) / f"{dataset_name}.npz"
            if file.exists():
                x = np.load(file)["x"]
                y = np.load(file)["y"]
                return x, y
            else:
                x, y = fn(*args, **kwargs)
                np.savez(Path(root) / f"{dataset_name}.npz", x=x, y=y)
                return x, y

        return wrapper

    return decorator


def get_uci_dataset(
    id: int, targets: Literal["binary", "multiclass", "regression", "raw"] = "binary"
) -> tuple[np.ndarray, np.ndarray]:
    """Obtain a dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=id)

    x = dataset.data.features
    x = categorical_to_int(x)
    x = x.to_numpy().astype(np.float32)
    x = np.nan_to_num(x)

    y = dataset.data.targets
    match targets:
        case "binary":
            y = y.to_numpy().astype(np.bool_)
        case "multiclass":
            y = y.to_numpy().astype(np.int32).squeeze()
        case "regression":
            y = y.to_numpy().astype(np.float32)

    return x, y


def german_credit() -> tuple[np.ndarray, np.ndarray]:
    """German Credit Dataset from the UCI ML Repository."""
    x, y = get_uci_dataset(id=144, targets="raw")
    y = (y > 1).to_numpy().astype(np.bool_)
    return x, y


def german() -> tuple[np.ndarray, np.ndarray]:
    return german_credit()


def obesity() -> tuple[np.ndarray, np.ndarray]:
    """Obesity Levels Based On Eating Habits and Physical Condition Dataset
    from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=544)

    x = dataset.data.features
    x.loc[:, "Gender"] = x["Gender"].map({"Male": 0, "Female": 1})
    x.loc[:, "family_history_with_overweight"] = x[
        "family_history_with_overweight"
    ].map({"yes": 1, "no": 0})
    x.loc[:, "FAVC"] = x["FAVC"].map({"yes": 1, "no": 0})
    x.loc[:, "CAEC"] = x["CAEC"].map(
        {"Sometimes": 1, "Frequently": 2, "Always": 3, "no": 0}
    )
    x.loc[:, "SMOKE"] = x["SMOKE"].map({"yes": 1, "no": 0})
    x.loc[:, "SCC"] = x["SCC"].map({"yes": 1, "no": 0})
    x.loc[:, "CALC"] = x["CALC"].map(
        {"Sometimes": 1, "Frequently": 2, "Always": 3, "no": 0}
    )
    x.loc[:, "MTRANS"] = x["MTRANS"].map(
        {
            "Automobile": 0,
            "Public_Transportation": 1,
            "Motorbike": 2,
            "Bike": 3,
            "Walking": 4,
        }
    )
    x = x.to_numpy().astype(np.float32)

    y = dataset.data.targets
    y = y["NObeyesdad"].map(
        {
            "Insufficient_Weight": 0,
            "Normal_Weight": 1,
            "Overweight_Level_I": 2,
            "Overweight_Level_II": 3,
            "Obesity_Type_I": 4,
            "Obesity_Type_II": 5,
            "Obesity_Type_III": 6,
        }
    )
    y = y.to_numpy().astype(np.int32)

    return x, y


def vehicles() -> tuple[np.ndarray, np.ndarray]:
    """Vehicle class from shape dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=149)

    x = dataset.data.features
    x.loc[:, "COMPACTNESS"] = x["COMPACTNESS"].fillna(x["COMPACTNESS"].mean())
    x = x.to_numpy().astype(np.float32)

    y = dataset.data.targets
    y = y["class"].map(
        {
            "van": 0,
            "saab": 1,
            "bus": 2,
            "opel": 3,
            "204": 0,  # there is just one sample of this class
        }
    )
    y = y.to_numpy().astype(np.int32)

    return x, y


def parkinsons() -> tuple[np.ndarray, np.ndarray]:
    """Parkinsons Telemonitoring dataset from the UCI ML Repository."""
    return get_uci_dataset(id=189, targets="regression")


@save_dataset("cdc_diabetes")
def cdc_diabetes() -> tuple[np.ndarray, np.ndarray]:
    """Appartment for Rent Dataset from the UCI ML Repository."""
    return get_uci_dataset(id=891, targets="binary")


def mushroom() -> tuple[np.ndarray, np.ndarray]:
    """Mushroom Dataset from the UCI ML Repository.

    Predict whether a mushroom is edible or poisonous.
    """
    x, y = get_uci_dataset(id=73, targets="raw")
    y = (y == "p").to_numpy().astype(np.bool_)
    return x, y


def default() -> tuple[np.ndarray, np.ndarray]:
    """Default of Credit Card Clients Dataset from the UCI ML Repository."""
    return get_uci_dataset(id=350, targets="binary")


def automobile() -> tuple[np.ndarray, np.ndarray]:
    """Automobile Dataset from the UCI ML Repository."""
    x, y = get_uci_dataset(id=10, targets="raw")
    y = y["symboling"]
    y = y.to_numpy().astype(np.float32)  # make this a regression task
    return x, y


def steel() -> tuple[np.ndarray, np.ndarray]:
    """Steel Plates Faults Dataset from the UCI ML Repository."""
    return get_uci_dataset(id=198, targets="regression")


@save_dataset("uci_appliances_energy_prediction")
def appliances() -> tuple[np.ndarray, np.ndarray]:
    """Appliances Energy Prediction Dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=374)

    x = dataset.data.features
    x.loc[:, "date"] = x["date"].str[-8:].str.replace(":", "").astype(int)
    x = x.to_numpy().astype(np.float32)

    y = dataset.data.targets
    y = y.to_numpy().astype(np.float32)  # make this a regression task

    return x, y


def hepatitis() -> tuple[np.ndarray, np.ndarray]:
    """Hepatitis C Virus (HCV) for Egyptian patients Dataset from
    the UCI ML Repository.
    """
    x, y = get_uci_dataset(id=503, targets="multiclass")
    y = y - 1
    return x, y


def breast_cancer() -> tuple[np.ndarray, np.ndarray]:
    """Breast Cancer Dataset from the UCI ML Repository."""
    x, y = get_uci_dataset(id=17, targets="raw")
    y = (y == "M").to_numpy().astype(np.bool_)
    return x, y


def infrared_temperature() -> tuple[np.ndarray, np.ndarray]:
    """Infrared Thermography Temperature Dataset from the UCI ML Repository."""
    return get_uci_dataset(id=925, targets="regression")


def ionosphere() -> tuple[np.ndarray, np.ndarray]:
    """Ionosphere Dataset from the UCI ML Repository."""
    x, y = get_uci_dataset(id=52, targets="raw")
    y = (y == "g").to_numpy().astype(np.bool_)
    return x, y


def dropout() -> tuple[np.ndarray, np.ndarray]:
    """Student Dropout and Academic Success Dataset from the UCI ML Repository."""
    x, y = get_uci_dataset(id=697, targets="raw")
    y = (y == "Dropout").to_numpy().astype(np.bool_)  # make binary classification task
    return x, y


def annealing() -> tuple[np.ndarray, np.ndarray]:
    """Annealing Dataset from the UCI ML Repository."""
    x, y = get_uci_dataset(id=3, targets="raw")
    y = (y != "3").to_numpy().astype(np.bool_)  # make binary classification task
    return x, y


@save_dataset("uci_census_income_kdd")
def support2() -> tuple[np.ndarray, np.ndarray]:
    """Support2 Dataset from the UCI ML Repository."""
    x, y = get_uci_dataset(id=880, targets="raw")
    y = y["death"].to_numpy().astype(np.bool_)
    return x, y


@save_dataset("uci_diabetes130")
def diabetes130() -> tuple[np.ndarray, np.ndarray]:
    """Diabetes j130 Dataset from the UCI ML Repository."""
    x, y = get_uci_dataset(id=296, targets="raw")
    y = (y != "NO").to_numpy().astype(np.bool_)  # make binary classification task
    return x, y


@save_dataset("uci_covertype")
def covertype() -> tuple[np.ndarray, np.ndarray]:
    """Covertype Dataset from the UCI ML Repository."""
    x, y = get_uci_dataset(id=31, targets="raw")
    y = categorical_to_int(y) - 1
    y = y.to_numpy().astype(np.int32).squeeze()
    return x, y


def lung_cancer() -> tuple[np.ndarray, np.ndarray]:
    """Lung Cancer Dataset from the UCI ML Repository."""
    x, y = get_uci_dataset(id=62, targets="raw")
    y = categorical_to_int(y) - 1
    y = y.to_numpy().astype(np.int32).squeeze()
    return x, y


def spambase() -> tuple[np.ndarray, np.ndarray]:
    """Spambase Dataset from the UCI ML Repository."""
    return get_uci_dataset(id=94, targets="binary")


@save_dataset("uci_online_news_popularity")
def online_news() -> tuple[np.ndarray, np.ndarray]:
    """Online News Popularity Dataset from the UCI ML Repository."""
    x, y = get_uci_dataset(id=332, targets="raw")
    # the regression task is too hard, make this a binary classification task
    y = (y > y.median()).to_numpy().astype(np.bool_)
    return x, y


def sonar() -> tuple[np.ndarray, np.ndarray]:
    """Connectionist Bench (Sonar, Mines vs. Rocks) Dataset from the UCI ML Repository."""
    return get_uci_dataset(id=151, targets="binary")


def handwritten_digits() -> tuple[np.ndarray, np.ndarray]:
    """Handwritten Digits Dataset from the UCI ML Repository."""
    return get_uci_dataset(id=80, targets="multiclass")


@save_dataset("uci_rt_iot2022")
def rt_iot() -> tuple[np.ndarray, np.ndarray]:
    """RT-IoT 2022 Dataset from the UCI ML Repository."""
    x, y = get_uci_dataset(id=942, targets="raw")
    y = categorical_to_int(y) - 1
    y = y.to_numpy().astype(np.int32).squeeze()
    return x, y


def bankruptcy() -> tuple[np.ndarray, np.ndarray]:
    """Taiwanese Bankruptcy Dataset from the UCI ML Repository."""
    return get_uci_dataset(id=572, targets="binary")


# ==============================================================================
# NIH Chest X-ray Dataset
# ==============================================================================


class NIHChestXrayDataset(Dataset):
    """PyTorch dataset for NIH Chest X-ray using data downloaded via kagglehub.

    Args:
        split: One of ``"train"``, ``"val"``, ``"trainval"``, ``"test"``, or
            ``"all"``. Validation is carved deterministically from the official
            train/val list using ``val_fraction``.
        transform: Optional torchvision-style transform applied to the PIL image.
        target_transform: Optional transform applied to the multi-hot label tensor.
        val_fraction: Fraction of the official train/val split reserved for
            validation when ``split`` is ``"train"`` or ``"val"``.
        seed: Deterministic seed used when partitioning the train/val list.
        download: If ``True`` and ``root`` is not provided, resolve the dataset
            via ``kagglehub.dataset_download`` (may download if not cached).
    """

    NIH_CHEST_XRAY_LABELS: Sequence[str] = (
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Hernia",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pleural_Thickening",
        "Pneumonia",
        "Pneumothorax",
        "No Finding",
    )

    def __init__(
        self,
        split: str = "train",
        transform=None,
        target_transform=None,
        val_fraction: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.root = Path(kagglehub.dataset_download("nih-chest-xrays/data"))
        self.transform = transform
        self.target_transform = target_transform

        self.metadata = self._load_metadata()
        split_names = self._resolve_split(split, val_fraction, seed)
        self.samples = self.metadata[
            self.metadata["Image Index"].isin(split_names)
        ].reset_index(drop=True)

        self._image_lookup = self._index_images(self.samples["Image Index"])
        if self.samples.empty:
            raise RuntimeError(
                "No samples found for split "
                f"{split!r}. Check that the dataset is extracted correctly."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples.iloc[idx]
        image_path = self._image_lookup.get(row["Image Index"])
        if image_path is None:
            raise FileNotFoundError(
                f"Could not find image {row['Image Index']} on disk"
            )

        image = Image.open(image_path).convert("RGB")
        target = self._encode_labels(row["Finding Labels"])

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def _load_metadata(self) -> pd.DataFrame:
        candidates = [
            self.root / "Data_Entry_2017.csv",
            self.root / "Data_Entry_2017_v2020.csv",
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
        if csv_path is None:
            raise FileNotFoundError(
                f"Could not find dataset metadata CSV in {self.root}. "
                "Ensure kagglehub.dataset_download('nih-chest-xrays/data') "
                "has been extracted."
            )
        metadata = pd.read_csv(csv_path)
        expected_columns = {"Image Index", "Finding Labels"}
        if not expected_columns.issubset(metadata.columns):
            raise ValueError(
                f"Metadata CSV missing expected columns {expected_columns}. "
                f"Found columns: {set(metadata.columns)}"
            )
        return metadata[["Image Index", "Finding Labels"]]

    def _resolve_split(self, split: str, val_fraction: float, seed: int) -> set[str]:
        split = split.lower()
        if split == "all":
            return set(self.metadata["Image Index"])

        if split == "test":
            test_file = self.root / "test_list.txt"
            return self._load_split_file(test_file)

        train_val_file = self.root / "train_val_list.txt"
        train_val_names = self._load_split_file(train_val_file)
        if split == "trainval":
            return train_val_names

        # Deterministic partition of the official train/val list.
        rng = np.random.default_rng(seed)
        train_val_names = list(train_val_names)
        permuted = [train_val_names[i] for i in rng.permutation(len(train_val_names))]
        split_idx = int(len(permuted) * (1 - val_fraction))
        train_names = set(permuted[:split_idx])
        val_names = set(permuted[split_idx:])

        if split == "train":
            return train_names
        elif split == "val":
            return val_names

        raise ValueError(f"Unknown split {split!r}.")

    def _index_images(self, requested_files: Iterable[str]) -> dict[str, Path]:
        """Index image paths for the requested filenames only."""
        requested = set(requested_files)
        image_roots = self._find_image_roots(self.root)
        if not image_roots:
            raise FileNotFoundError(
                f"Could not find any image folders under {self.root}. "
                "The dataset should contain directories like images_001/images."
            )

        lookup: dict[str, Path] = {}
        for root in image_roots:
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for img_path in root.glob(ext):
                    name = img_path.name
                    if name in requested:
                        lookup[name] = img_path
                        if len(lookup) == len(requested):
                            return lookup
        return lookup

    def _encode_labels(self, label_string: str) -> torch.Tensor:
        target = torch.zeros(len(self.NIH_CHEST_XRAY_LABELS), dtype=torch.float32)
        labels = [label.strip() for label in label_string.split("|")]
        if labels == [""]:
            return target
        for label in labels:
            try:
                idx = self.NIH_CHEST_XRAY_LABELS.index(label)
            except ValueError as ex:
                raise ValueError(
                    f"Unknown label '{label}' in NIH ChestX-ray metadata"
                ) from ex
            target[idx] = 1.0
        return target

    @staticmethod
    def _find_image_roots(root: Path) -> list[Path]:
        """Locate folders that contain the actual PNG/JPEG files."""
        candidates = [path for path in root.glob("**/images") if path.is_dir()]
        # Some archives extract directly to images_001, images_002, ...
        candidates.extend([path for path in root.glob("images_*") if path.is_dir()])
        candidates.extend([root / "images"] if (root / "images").is_dir() else [])
        # Deduplicate while keeping stable order.
        seen = set()
        unique: list[Path] = []
        for path in candidates:
            if path not in seen:
                unique.append(path)
                seen.add(path)
        return unique

    @staticmethod
    def _load_split_file(path: Path) -> set[str]:
        if not path.exists():
            raise FileNotFoundError(
                f"Expected split file {path} not found. "
                "Please ensure the dataset archives are extracted."
            )
        with path.open("r") as f:
            names = {line.strip() for line in f if line.strip()}
        return names
