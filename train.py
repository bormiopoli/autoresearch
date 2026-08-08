# `train.py`

"""
MUTABLE RESEARCH FILE

This is the ONLY file that the research agent should modify.

The evaluation harness is in prepare.py and must remain immutable.

The agent should change one hypothesis at a time, run this file,
inspect the final Sharpe, and keep/revert the change.
"""

from __future__ import annotations

import time
from typing import Any

import tensorflow as tf
import keras

import prepare

# =====================================================================

# EXPERIMENT CONFIGURATION

# =====================================================================

#

# The research agent is encouraged to modify these values.

#

# Start from a stable baseline and make one change per experiment.

#

SEED = 42

# ---------------------------------------------------------------------

# Architecture

# ---------------------------------------------------------------------

GATE = True

HIDDEN_DIM = 128

DIM_TEMPORAL_REGIME = 96

DIM_TEMPORAL_VOLUME = 96

HEAD_NUM = 4

KERNEL_SIZE = 5

# ---------------------------------------------------------------------

# Temporal architecture

# ---------------------------------------------------------------------

TEMPORAL_ATTENTION_TYPE = "LSTM"

USE_LSTM = True

# ---------------------------------------------------------------------

# Regularisation

# ---------------------------------------------------------------------

DROPOUT = 0.30

W_DECAY = 1e-3

# ---------------------------------------------------------------------

# Data / temporal context

# ---------------------------------------------------------------------

STANDARDISATION = "standard"

WINDOW = 30

# ---------------------------------------------------------------------

# Portfolio transformation

# ---------------------------------------------------------------------

SHARPNESS = 15

TAU = 0.15

DEADZONE = 0.02

# ---------------------------------------------------------------------

# Optimisation

# ---------------------------------------------------------------------

INITIAL_LR = 1e-3

NUM_EPOCHS = 40

COMPUTING_BATCH = 192

# ---------------------------------------------------------------------

# Activations

# ---------------------------------------------------------------------

ACTIVATION = "gelu"

GATE_ACTIVATION = "tanh"

TEMPORAL_ACTIVATION = "gelu"

OUTPUT_ACTIVATION = "tanh"

# ---------------------------------------------------------------------

# Auxiliary supervision

# ---------------------------------------------------------------------

BLEND_SUPERVISION_WEIGHT = 0.25

# =====================================================================

# AUTORESEARCH TIME BUDGET

# =====================================================================

#

# Set to None to use NUM_EPOCHS.

#

# If you want the closest analogue to Karpathy-style fixed-compute

# research, set this to a fixed wall-clock budget.

#

# Example:

#

# TIME_BUDGET_SECONDS = 15 * 60

#

# All experiments then receive approximately the same compute budget.

#

TIME_BUDGET_SECONDS = None

# =====================================================================

# PARAMETER DICTIONARY

# =====================================================================

def get_params() -> dict[str, Any]:
    """
    Return the complete mutable experiment configuration.

    ```
    This dictionary is passed to the original model factory.
    """

    return {
        "seed": SEED,

        "gate": GATE,

        "num_epochs": NUM_EPOCHS,

        "computing_batch": COMPUTING_BATCH,

        "dropout": DROPOUT,

        "standardisation": STANDARDISATION,

        "standardisation_type_volume": STANDARDISATION,

        "temporal_attention_type": TEMPORAL_ATTENTION_TYPE,

        "use_lstm": USE_LSTM,

        "hidden_dim": HIDDEN_DIM,

        "dim_temporal_regime": DIM_TEMPORAL_REGIME,

        "dim_temporal_volume": DIM_TEMPORAL_VOLUME,

        "head_num": HEAD_NUM,

        "kernel_size": KERNEL_SIZE,

        "w_decay": W_DECAY,

        "window": WINDOW,

        "sharpness": SHARPNESS,

        "tau": TAU,

        "initial_lr": INITIAL_LR,

        "activation": ACTIVATION,

        "gate_activation": GATE_ACTIVATION,

        "temporal_activation": TEMPORAL_ACTIVATION,

        "output_activation": OUTPUT_ACTIVATION,

        "blend_supervision_weight": BLEND_SUPERVISION_WEIGHT,

        "deadzone": DEADZONE,
    }


# =====================================================================

# OPTIONAL TIME-LIMITED CALLBACK

# =====================================================================

class TimeBudgetCallback(keras.callbacks.Callback):
    """
    Stop training once the fixed wall-clock budget is reached.

    ```
    This is preferable to simply comparing epoch counts if experiments
    have different computational costs.
    """

    def __init__(self, seconds: float):
        super().__init__()

        self.seconds = float(seconds)

        self.start_time = None

        self.epochs_completed = 0

    def on_train_begin(self, logs=None):
        self.start_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):

        self.epochs_completed = epoch + 1

        elapsed = time.perf_counter() - self.start_time

        if elapsed >= self.seconds:
            self.model.stop_training = True


# =====================================================================

# TRAINING

# =====================================================================

def train():


    prepare.assert_research_mode()

    params = get_params()

    prepare.set_seed(SEED)

    prepare.clear_session()

    experiment_start = time.perf_counter()

    # ---------------------------------------------------------------
    # Fixed data
    # ---------------------------------------------------------------

    data = prepare.prepare_split(
        standardisation=STANDARDISATION,
        window=WINDOW,
    )

    prepare.describe_split(data)

    print("EXPERIMENT")
    print("----------")

    for key, value in params.items():
        print(f"{key}: {value}")

    print()

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------

    train_ds = prepare.make_dataset(
        data,
        batch_size=COMPUTING_BATCH,
        validation=False,
    )

    val_ds = prepare.make_dataset(
        data,
        batch_size=COMPUTING_BATCH,
        validation=True,
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    model = prepare.build_model(
        params,
        data,
    )

    print()
    print("MODEL")
    print("-----")

    model.summary()

    # ---------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------

    callbacks = []

    # Reuse the original callback factory.
    #
    # This preserves the existing optimizer/LR machinery from your
    # implementation.
    try:

        callback_builder = getattr(
            prepare.core,
            "_build_callbacks",
        )

        original_callbacks, _ = callback_builder(
            params=params,
            n_train_samples=data["x"].shape[0],
            num_epochs=NUM_EPOCHS,
            model=model,
            use_early_stopping=False,
        )

        callbacks.extend(original_callbacks)

    except Exception as exc:

        print(
            "WARNING: original callback builder could not be used: "
            f"{exc}"
        )

    # ---------------------------------------------------------------
    # Optional fixed wall-clock budget
    # ---------------------------------------------------------------

    time_callback = None

    if TIME_BUDGET_SECONDS is not None:

        time_callback = TimeBudgetCallback(
            TIME_BUDGET_SECONDS
        )

        callbacks.append(time_callback)

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    print()
    print("TRAINING")
    print("--------")

    fit_start = time.perf_counter()

    model.fit(
        train_ds,
        validation_data=None,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
        verbose=0,
    )

    training_elapsed = (
        time.perf_counter() - fit_start
    )

    if time_callback is not None:
        epochs_completed = time_callback.epochs_completed
    else:
        epochs_completed = NUM_EPOCHS

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    print()
    print("VALIDATION")
    print("----------")

    metric = prepare.evaluate_validation(
        model,
        data,
    )

    total_elapsed = (
        time.perf_counter() - experiment_start
    )

    print()
    print(
        f"Validation Sharpe: {metric['sharpe']:.8f}"
    )

    print(
        f"Validation profit: {metric['profit']:.8f}"
    )

    print(
        f"Training time: {training_elapsed:.3f}s"
    )

    print(
        f"Total time: {total_elapsed:.3f}s"
    )

    # ---------------------------------------------------------------
    # Machine-readable output
    # ---------------------------------------------------------------

    prepare.print_result(
        metric=metric,
        elapsed=total_elapsed,
        epochs_completed=epochs_completed,
        params=params,
    )

    # ---------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------

    del model
    del train_ds
    del val_ds
    del data

    prepare.clear_session()

    return metric


# =====================================================================

# ENTRY POINT

# =====================================================================

if __name__ == "__main__":

    try:

        train()

    except KeyboardInterrupt:

        print()
        print("Experiment interrupted.")

        raise

    except Exception as exc:

        print()
        print("---")
        print("status: failed")
        print(
            f"error: {type(exc).__name__}: {exc}"
        )

        raise
