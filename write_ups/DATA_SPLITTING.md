# Data Splitting

768 people total.

## Outer split: 4-fold cross-validation

All 768 people are divided into 4 groups of 192. Each round, 3 groups (576 people) train the model and 1 group (192 people) is the test set. This rotates 4 times so every person gets tested exactly once.

The split is stratified on class label and whether the person has any missing features. Every fold has the same class ratio (65/35) and the same missingness rate (49%).

## Inner split: train/validation

Within each fold's 576 training people, an 80/20 split creates 461 inner-train and 115 validation.

## What each set does

- Inner-train (461): Train a temporary model to try different hyperparameter combinations.
- Validation (115): Evaluate which combination works best. Pick the winner.
- After tuning, retrain a fresh model on all 576 training people with the selected hyperparameters.
- Test (192): Final evaluation. Never seen during tuning.
