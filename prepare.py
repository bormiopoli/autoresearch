"""
IMMUTABLE EXPERIMENT HARNESS

This file defines the fixed research environment for Autoresearch.

DO NOT MODIFY THIS FILE DURING RESEARCH.

The mutable research file is train.py.

The underlying model/data implementation lives in:
production_crossectional.py

This wrapper intentionally reuses the existing functions from that file:
prepare_features_from_wide_df2
_prepare_arrays
_build_inputs_dict
_model_kwargs
_build_callbacks
build_cross_sectional_mixer_lstm2
make_dataset
backtest_weights
compute_horizon_quality

The original code already uses a walk-forward structure in which the
validation period is kept separate from the training period. The original
FWCV implementation also evaluates predictions using backtest_weights()
and uses -Sharpe as the minimized objective.
"""

from __future__ import annotations

import gc
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf
import keras

# ---------------------------------------------------------------------
# IMPORTANT:
# production_crossectional.py is your existing 8497-line implementation.
# ---------------------------------------------------------------------

import production_crossectional as core


# =====================================================================
# IMMUTABLE RESEARCH CONFIGURATION
# =====================================================================

SEED = 42

# Number of bars used for the fixed research validation.
#
# If None, the last 20% of the research dataset is used.
VALIDATION_BARS = None

# Gap between training and validation.
# This should remain fixed throughout the research campaign.
VALIDATION_GAP = 1

# The research loop must never access the true OOS period.
USE_TRUE_OOS = False


# =====================================================================
# DATA ACCESS
# =====================================================================

def _get(name: str) -> Any:
    """
    Retrieve an object from the original production module.

    This keeps prepare.py independent from the exact way the original
    notebook creates the global objects.
    """
    if not hasattr(core, name):
        raise AttributeError(
            f"production_crossectional.py does not expose '{name}'. "
            f"Make sure the original notebook code is saved as "
            f"production_crossectional.py."
        )

    return getattr(core, name)


def get_data():
    """
    Return the immutable research dataset.

    The true OOS dataset is intentionally NOT returned.
    """
    data = _get("initial_data_full")
    br_data = _get("br_data")
    macro_df = _get("macro_df")

    tickers = _get("tickers")

    selected_cols = _get("selected_cols")
    selected_regime_cols = _get("selected_regime_cols")
    selected_macro_cols = _get("selected_macro_cols")

    return {
        "data": data,
        "br_data": br_data,
        "macro_df": macro_df,
        "tickers": tickers,
        "selected_cols": selected_cols,
        "selected_regime_cols": selected_regime_cols,
        "selected_macro_cols": selected_macro_cols,
    }


# =====================================================================
# DETERMINISM
# =====================================================================

def set_seed(seed: int = SEED) -> None:
    """
    Set all relevant random seeds.

    The original implementation already uses deterministic TensorFlow
    operations. We additionally set Python / NumPy / TensorFlow seeds
    here so that experiments are reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        keras.utils.set_random_seed(seed)
    except Exception:
        pass


def clear_session() -> None:
    """
    Release TensorFlow/Keras state between experiments.
    """
    try:
        keras.backend.clear_session()
    except Exception:
        pass

    gc.collect()


# =====================================================================
# FIXED TRAIN / VALIDATION SPLIT
# =====================================================================

@dataclass(frozen=True)
class Split:
    train_start: int
    train_end: int
    val_start: int
    val_end: int


def make_fixed_split(
    n_bars: int,
    validation_bars: int | None = VALIDATION_BARS,
    gap: int = VALIDATION_GAP,
) -> Split:
    """
    Create exactly one immutable chronological validation split.

    No random splitting.
    No expanding-window choice.
    No tuning of the split.

    The last validation period is reserved for evaluating experiments.
    """
    if validation_bars is None:
        validation_bars = max(60, n_bars // 5)

    if validation_bars <= 0:
        raise ValueError("validation_bars must be positive")

    if gap < 0:
        raise ValueError("gap must be >= 0")

    val_end = n_bars
    val_start = val_end - validation_bars

    train_end = val_start - gap
    train_start = 0

    if train_end <= train_start:
        raise ValueError(
            f"Invalid split: n_bars={n_bars}, "
            f"validation_bars={validation_bars}, gap={gap}"
        )

    return Split(
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
    )


# =====================================================================
# FEATURE PREPARATION
# =====================================================================

def prepare_split(
    *,
    standardisation: str,
    window: int,
    path_suffix: str = "",
):
    """
    Prepare the fixed train/validation split.

    The validation data is prepared independently with training=False.

    This function intentionally uses the exact feature-preparation
    implementation from the original model code.
    """
    objects = get_data()

    data = objects["data"]
    br_data = objects["br_data"]
    macro_df = objects["macro_df"]

    selected_cols = objects["selected_cols"]
    selected_regime_cols = objects["selected_regime_cols"]
    selected_macro_cols = objects["selected_macro_cols"]

    split = make_fixed_split(len(data))

    train_df = data.iloc[split.train_start:split.train_end]
    train_br = br_data.iloc[split.train_start:split.train_end]
    train_macro = macro_df.iloc[split.train_start:split.train_end]

    val_df = data.iloc[split.val_start:split.val_end]
    val_br = br_data.iloc[split.val_start:split.val_end]
    val_macro = macro_df.iloc[split.val_start:split.val_end]

    prepare_features = _get("prepare_features_from_wide_df2")

    # ---------------------------------------------------------------
    # Training features
    # ---------------------------------------------------------------

    (
        x,
        x_reg_onchain,
        x_reg_macro,
        y,
        y_r,
        vol,
        r1,
        r3,
        r7,
        r10,
        r13,
        r17,
        r24,
    ) = prepare_features(
        df=train_df,
        btc_chain_df=train_br,
        macro_df=train_macro,
        standardisation_type=standardisation,
        training=True,
        path_suffix=path_suffix,
        selected_cols=selected_cols,
        selected_regime_cols=selected_regime_cols,
        selected_macro_cols=selected_macro_cols,
        standardisation_type_volume=standardisation,
    )

    # ---------------------------------------------------------------
    # Validation features
    # ---------------------------------------------------------------

    (
        x_val,
        x_val_reg_onchain,
        x_val_reg_macro,
        y_val,
        y_valr,
        vol_val,
        r1_val,
        r3_val,
        r7_val,
        r10_val,
        r13_val,
        r17_val,
        r24_val,
    ) = prepare_features(
        df=val_df,
        btc_chain_df=val_br,
        macro_df=val_macro,
        standardisation_type=standardisation,
        training=False,
        path_suffix=path_suffix,
        selected_cols=selected_cols,
        selected_regime_cols=selected_regime_cols,
        selected_macro_cols=selected_macro_cols,
        standardisation_type_volume=standardisation,
    )

    # ---------------------------------------------------------------
    # Windowing
    # ---------------------------------------------------------------

    prepare_arrays = _get("_prepare_arrays")

    (
        x,
        x_reg_onchain,
        x_reg_macro,
        y_r,
        y_shifted,
        y_shifted_market,
        x_w,
        y_shifted_w,
        y_shifted_market_w,
        vol_w,
    ) = prepare_arrays(
        x,
        x_reg_onchain,
        x_reg_macro,
        y,
        y_r,
        vol,
        window,
    )

    (
        x_val,
        x_val_reg_onchain,
        x_val_reg_macro,
        y_valr_window,
        y_shifted_val,
        y_shifted_market_val,
        x_w_val,
        y_shifted_w_val,
        y_shifted_market_w_val,
        vol_val_w,
    ) = prepare_arrays(
        x_val,
        x_val_reg_onchain,
        x_val_reg_macro,
        y_val,
        y_valr,
        vol_val,
        window,
    )

    return {
        "split": split,

        "x": x,
        "x_reg_onchain": x_reg_onchain,
        "x_reg_macro": x_reg_macro,
        "y": y,
        "y_r": y_r,

        "r1": r1,
        "r3": r3,
        "r7": r7,
        "r10": r10,
        "r13": r13,
        "r17": r17,
        "r24": r24,

        "x_val": x_val,
        "x_val_reg_onchain": x_val_reg_onchain,
        "x_val_reg_macro": x_val_reg_macro,
        "y_val": y_val,
        "y_valr": y_valr,

        "r1_val": r1_val,
        "r3_val": r3_val,
        "r7_val": r7_val,
        "r10_val": r10_val,
        "r13_val": r13_val,
        "r17_val": r17_val,
        "r24_val": r24_val,

        "y_shifted": y_shifted,
        "y_shifted_market": y_shifted_market,
        "x_w": x_w,
        "y_shifted_w": y_shifted_w,
        "y_shifted_market_w": y_shifted_market_w,
        "vol_w": vol_w,

        "y_shifted_val": y_shifted_val,
        "y_shifted_market_val": y_shifted_market_val,
        "x_w_val": x_w_val,
        "y_shifted_w_val": y_shifted_w_val,
        "y_shifted_market_w_val": y_shifted_market_w_val,
        "vol_val_w": vol_val_w,
    }


# =====================================================================
# DATASET CONSTRUCTION
# =====================================================================

def build_inputs(d):
    """
    Build exactly the same input dictionary used by the original model.
    """
    build_inputs_dict = _get("_build_inputs_dict")

    return build_inputs_dict(
        d["x"],
        d["x_reg_onchain"],
        d["x_reg_macro"],
        d["y_shifted"],
        d["y_shifted_market"],
        d["x_w"],
        d["y_shifted_w"],
        d["y_shifted_market_w"],
        d["vol_w"],
    )


def build_validation_inputs(d):
    """
    Build the validation input dictionary.
    """
    build_inputs_dict = _get("_build_inputs_dict")

    return build_inputs_dict(
        d["x_val"],
        d["x_val_reg_onchain"],
        d["x_val_reg_macro"],
        d["y_shifted_val"],
        d["y_shifted_market_val"],
        d["x_w_val"],
        d["y_shifted_w_val"],
        d["y_shifted_market_w_val"],
        d["vol_val_w"],
    )


def build_targets(d, validation: bool = False):
    """
    Construct the multi-output targets expected by the model.
    """
    compute_horizon_quality = _get("compute_horizon_quality")

    if validation:
        r = [
            d["r1_val"],
            d["r3_val"],
            d["r7_val"],
            d["r10_val"],
            d["r13_val"],
            d["r17_val"],
            d["r24_val"],
        ]

        quality = compute_horizon_quality(
            r,
            exposures_list=None,
        )

        n = len(d["x_val"])

        return {
            "exposures": d["y_val"][-n:],
            "exposures_h1": d["r1_val"][-n:],
            "exposures_h3": d["r3_val"][-n:],
            "exposures_h7": d["r7_val"][-n:],
            "exposures_h10": d["r10_val"][-n:],
            "exposures_h13": d["r13_val"][-n:],
            "exposures_h17": d["r17_val"][-n:],
            "exposures_h24": d["r24_val"][-n:],
            "blend_weights_target": quality[-n:],
        }

    r = [
        d["r1"],
        d["r3"],
        d["r7"],
        d["r10"],
        d["r13"],
        d["r17"],
        d["r24"],
    ]

    quality = compute_horizon_quality(
        r,
        exposures_list=None,
    )

    n = len(d["x"])

    return {
        "exposures": d["y_r"][-n:],
        "exposures_h1": d["r1"][-n:],
        "exposures_h3": d["r3"][-n:],
        "exposures_h7": d["r7"][-n:],
        "exposures_h10": d["r10"][-n:],
        "exposures_h13": d["r13"][-n:],
        "exposures_h17": d["r17"][-n:],
        "exposures_h24": d["r24"][-n:],
        "blend_weights_target": quality[-n:],
    }


def make_dataset(
    d,
    batch_size: int,
    validation: bool = False,
):
    """
    Construct a tf.data.Dataset.
    """
    dataset_builder = _get("make_dataset")

    if validation:
        inputs = build_validation_inputs(d)
    else:
        inputs = build_inputs(d)

    targets = build_targets(
        d,
        validation=validation,
    )

    return dataset_builder(
        inputs_dict=inputs,
        targets_dict=targets,
        batch_size=batch_size,
        shuffle=False,
    )


# =====================================================================
# MODEL CONSTRUCTION
# =====================================================================

def build_model(params: dict[str, Any], d):
    """
    Build the model using exactly the original model factory.

    The parameters come from train.py.
    """
    model_builder = _get("build_cross_sectional_mixer_lstm2")
    model_kwargs = _get("_model_kwargs")

    objects = get_data()

    return model_builder(
        **model_kwargs(
            params,
            n_features=d["x"].shape[-1],
            n_onchain_features=d["x_reg_onchain"].shape[-1],
            n_macro_features=d["x_reg_macro"].shape[-1],
            n_assets=len(objects["tickers"]),
            n_vol_features=d["vol_w"].shape[-1],
        )
    )


# =====================================================================
# FIXED VALIDATION EVALUATION
# =====================================================================

def evaluate_validation(
    model,
    d,
):
    """
    Evaluate predictions exclusively on the fixed research-validation
    period.

    The metric is the same backtest Sharpe used in the original FWCV
    implementation.
    """
    predictions = model.predict(
        build_validation_inputs(d),
        verbose=0,
    )

    exposures = predictions["exposures"]

    y_val = d["y_val"]

    # Keep alignment exactly as in the original implementation.
    y_val = y_val[-len(exposures):]

    backtest_weights = _get("backtest_weights")

    equity, summary = backtest_weights(
        exposures,
        y_val,
    )

    sharpe = float(summary["sharpe"])
    profit = float(equity["equity"][-1])

    return {
        "sharpe": sharpe,
        "profit": profit,
        "summary": summary,
    }


# =====================================================================
# MACHINE-READABLE RESULT
# =====================================================================

def print_result(
    *,
    metric: dict[str, Any],
    elapsed: float,
    epochs_completed: int,
    params: dict[str, Any],
) -> None:
    """
    Print the final result in a stable format for an Autoresearch
    controller.
    """
    print()
    print("---")

    print(f"sharpe: {metric['sharpe']:.8f}")
    print(f"profit: {metric['profit']:.8f}")
    print(f"elapsed_seconds: {elapsed:.3f}")
    print(f"epochs_completed: {epochs_completed}")

    # Compact JSON representation of the experiment.
    serializable_params = {}

    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serializable_params[key] = value
        else:
            serializable_params[key] = str(value)

    print(
        "params_json: "
        + json.dumps(
            serializable_params,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


# =====================================================================
# SAFETY CHECKS
# =====================================================================

def assert_research_mode() -> None:
    """
    Prevent accidental use of the true OOS dataset.
    """
    if USE_TRUE_OOS:
        raise RuntimeError(
            "USE_TRUE_OOS=True is forbidden in the Autoresearch loop."
        )


def describe_split(d) -> None:
    """
    Print the fixed research split.
    """
    split = d["split"]

    print()
    print("FIXED RESEARCH SPLIT")
    print("--------------------")
    print(
        f"train: [{split.train_start}:{split.train_end}]"
    )
    print(
        f"validation: [{split.val_start}:{split.val_end}]"
    )
    print(
        f"train bars: {split.train_end - split.train_start}"
    )
    print(
        f"validation bars: {split.val_end - split.val_start}"
    )
    print()


# =====================================================================
# OPTIONAL DEBUG ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    assert_research_mode()

    objects = get_data()

    split = make_fixed_split(
        len(objects["data"])
    )

    print("prepare.py OK")
    print(f"Total bars: {len(objects['data'])}")
    print(
        f"Train: {split.train_start}:{split.train_end}"
    )
    print(
        f"Validation: {split.val_start}:{split.val_end}"
    )
