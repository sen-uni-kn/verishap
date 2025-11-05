# Copyright 2025 David Boetius
"""Utilities for parsing command line arguments for tabular datasets."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import equinox as eqx
import jax
import numpy as np

from ..argument_parser import CmdArgs
from .models import MLP
from .utils import load_dataset


class TabularCmdArgs(CmdArgs):
    def __init__(self, *parser_args, **parser_kwargs):
        super().__init__(*parser_args, **parser_kwargs)

    @property
    def dataset(self) -> tuple[np.ndarray, np.ndarray]:
        dataset = self.args.dataset
        if dataset is None:
            dataset = self.args.model.stem.split("-")[0]
        return load_dataset(dataset)

    @property
    def model(self) -> Callable:
        model = MLP.load(self.args.model)
        model = eqx.nn.inference_mode(model)
        model = jax.vmap(model, axis_name="batch")
        return model

    def out_file(self, local_output_dir: Path) -> Path:
        if self.args.out_file is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            out_file = (
                local_output_dir
                / f"{self.args.dataset}"
                / (
                    f"{self.args.model.stem}_{self.args.feature}_{self.args.output_feature}_shap"
                    f"_{self.method_name}_{self.args.shap_variant}_{timestamp}.csv"
                )
            )
        else:
            out_file = Path(self.args.out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        return out_file
