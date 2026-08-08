# Financial Autoresearch

## Goal

Improve the validation Sharpe of the cross-sectional asset model.

The objective is NOT to maximize training performance.

The only metric used to accept/reject an experiment is:

    validation Sharpe

Higher is better.

---

## Files

### prepare.py

IMMUTABLE.

Do not modify.

Contains:

- data preparation
- feature construction
- train/validation boundaries
- target construction
- evaluation
- backtesting
- Sharpe calculation
- transaction costs
- OOS definitions

### train.py

MUTABLE.

This is the only file you should modify. The syntax you use is python 3.10

You may modify:

- architecture
- hyperparameters
- optimizer
- learning-rate schedule
- regularization
- temporal encoder
- cross-sectional attention
- loss functions
- training procedure

Do not modify the evaluation metric.
Modify just what you need to change. Keep same code for the untouched part.

---

## Experiment protocol

1. Inspect the current train.py.
2. Form one concrete hypothesis.
3. Make the smallest useful change.
4. Run:

    python train.py

5. Read the final metrics.
6. Compare against the current best.
7. If improved:
   - keep the change
   - commit it

8. If not improved:
   - revert the change

9. Record the experiment.

Do NOT make multiple unrelated changes in one experiment.

---

## Research priorities

Prefer experiments in this order:

1. Cross-sectional attention
2. Temporal representation
3. Model capacity
4. Regularization
5. Optimization
6. Loss/objective
7. Feature interactions

---

## Important financial constraints

Never:

- use future information
- modify OOS data
- modify validation targets
- change the Sharpe calculation
- change transaction costs to improve the score
- leak future returns into features
- normalize using future observations
- tune against the final OOS period

---

## Search strategy

Avoid random hyperparameter sweeps.

Each experiment should have a hypothesis.

Bad:

> Try hidden_dim = 192.

Good:

> Increasing hidden_dim from 128 to 192 may improve the model's ability to represent cross-asset interactions introduced by the new attention block, without changing the temporal encoder.

---

## Research memory

Maintain a log of:

- experiment
- hypothesis
- changed parameter/code
- validation Sharpe
- delta vs best
- decision

Do not repeatedly try failed configurations unless a new hypothesis explains why they may now work.

---

## Promotion

The best candidates from the fast validation loop will periodically be evaluated with:

1. FWCV
2. seed robustness
3. true OOS
4. ensemble evaluation

A validation improvement is NOT considered a final research result until it survives the promotion tests.