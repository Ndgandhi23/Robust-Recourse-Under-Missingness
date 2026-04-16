# Evaluation

This document explains every metric used to evaluate the method — what it measures, why it's included, and what the results tell us.

**Context:** This project is about algorithmic recourse — when a model predicts someone as diabetic, we want to tell them what they'd need to change to be predicted as not diabetic. The twist: some features are missing, so the recourse must be robust to that uncertainty.

---

## Evaluation Pipeline

We follow the evaluation structure from ElliCE (Turbal et al., NeurIPS 2025), adapted to our setting which adds imputation uncertainty on top of model uncertainty.

**4-fold stratified cross-validation.** The dataset is split into 4 folds preserving class balance. Each fold trains a fresh model, Hessian, MICE imputer, bootstrap ensemble, and LOF model on its training split. All metrics are evaluated on the held-out test split. Results are reported as mean +/- standard error across folds.

**Hyperparameter tuning.** Within each fold, the training data is further split 80/20 into inner-train and validation. A temporary model is trained on inner-train, and a grid search over (epsilon, rho_coverage) is run on the validation denied individuals:
- Grid: epsilon in {0.0, 0.0005, 0.001, 0.005, 0.01} x rho_coverage in {0.85, 0.90, 0.95}
- Criterion: maximize feasibility rate, then robustness (against an ellipsoid evaluator at epsilon_target=0.005), then minimize cost
- After tuning, the final model is retrained on the full fold training data with the selected hyperparameters

Tuned hyperparameters per fold:
```
Fold 1: eps=0.0010  rho=0.85
Fold 2: eps=0.0010  rho=0.85
Fold 3: eps=0.0005  rho=0.85
Fold 4: eps=0.0010  rho=0.85
```

Three of four folds selected epsilon=0.001, one selected epsilon=0.0005 (achieving perfect validation robustness at a lower cost). All folds selected rho_coverage=0.85.

**Conditions.** We run four variants of the algorithm on the same population within each fold. The "ours" and "no-reveal" conditions use the tuned epsilon and rho_coverage. The ablations use fixed values.

| Condition | epsilon | rho_coverage | K_max | What it represents |
|---|---|---|---|---|
| **ours** | tuned | tuned | 3 | Full method: joint Rashomon + missingness robustness with reveals |
| **baseline** | 0.0 | tuned | 3 | Missingness robustness only, no Rashomon set |
| **no-robust** | 0.0 | tuned (rho forced to 0) | 3 | No robustness at all — point imputation, nominal recourse only |
| **no-reveal** | tuned | tuned | 0 | Full robustness but reveals are forbidden; edits only |

**Three evaluator types.** Each tests a different aspect of robustness:

| Evaluator | What it perturbs | What it fixes | Type |
|---|---|---|---|
| Bootstrap retrain (filtered) | theta (real retrained models) | imputation at mu | Empirical |
| Ellipsoid sampling | theta (synthetic from Hessian geometry) | imputation at mu | Empirical |
| AWP closed-form | theta AND x_miss (analytic worst case) | nothing — joint bound | Theoretical |

The first two test model robustness alone (how well recourse holds across alternative models). The third tests the joint theoretical guarantee over both model and imputation uncertainty.

---

## Unit 1 — Nominal Validity

**Why we use it:** This is the minimum correctness bar. Before worrying about robustness or retraining, we need to confirm the recommended action actually flips the model's decision. If this fails, the solver has a bug.

**What we do:** Apply the recommended action to the person's feature vector, fill any still-missing features with their imputed mean, and check whether the model's score clears the approval threshold.

**Results (4-fold CV):**

```
Condition     Nominal
ours          1.000 +/- 0.000
baseline      1.000 +/- 0.000
no-robust     1.000 +/- 0.000
no-reveal     1.000 +/- 0.000
```

**What it means:** Every feasible recourse recommendation flips the model's decision across all folds. This is a unit test — if it ever fails, stop and debug.

---

## Unit 2 — Model Retrain Validity

**Why we use it:** Nominal validity only checks against the one model we trained. The paper's core claim is that recourse should hold across the *space* of plausible models. Bootstrap retraining is an independent, empirical check of that claim — no theory required.

**What we do:** Retrain the model 50 times, each on a random resample of the training data. For each retrained model, check whether the recommended action still gets the person approved. Report the fraction that pass.

**Results (4-fold CV):**

```
Condition     Retrain Validity
ours          0.942 +/- 0.015
no-reveal     0.978 +/- 0.007
baseline      0.705 +/- 0.028
no-robust     0.505 +/- 0.018
```

**What it means:** The Rashomon term (tuned epsilon ~0.001) delivers a 24 percentage point improvement over baseline and a 44pp improvement over no-robust. Without robustness, roughly 1 in 2 retrained models reject the recourse. With it, nearly all hold. Standard errors are tight across folds — these differences are real.

---

## Unit 3 — AWP Validity (Sanity Check)

**Why we use it:** Unit 2 was empirical. This is the mathematical counterpart. The paper proves a closed-form lower bound (Proposition 2) that certifies the recourse holds for every model in the Rashomon ellipsoid and every plausible imputation simultaneously. This check confirms the bound is positive using the exact parameters each condition was optimized with.

**What we do:** Evaluate the Proposition 2 lower bound at each condition's own (epsilon, rho). If the bound is non-negative, the certificate holds.

**Results (4-fold CV):**

```
Condition     AWP Sanity
ours          1.000 +/- 0.000
baseline      1.000 +/- 0.000
no-robust     1.000 +/- 0.000
no-reveal     1.000 +/- 0.000
```

**What it means:** All conditions pass 100% at their own optimization parameters. This is expected and correct — each condition's solver ensures the bound holds at the parameters it was given. At epsilon=0 and rho=0, the bound reduces to the nominal score, which every feasible solution satisfies by construction. The real discrimination happens in the AWP robustness *curve* (Unit 8), where epsilon_eval is swept beyond the optimization value.

---

## Unit 4 — LOF Plausibility

**Why we use it:** A recourse action could be model-valid but biologically impossible — "increase your glucose to 400." The paper claims recourse should be plausible. LOF operationalizes that claim by checking whether the post-action person still looks like a realistic individual from the training data.

**What we do:** Fit a Local Outlier Factor model on the MICE-imputed training data (averaging 20 draws for stability). Score the post-action feature vector against that learned distribution. A score of 1.0 means the person is in a dense, realistic region. Higher values indicate increasing outlier-ness.

**Results (4-fold CV):**

```
Condition     LOF
ours          1.117 +/- 0.012
baseline      1.141 +/- 0.019
no-robust     1.136 +/- 0.020
no-reveal     1.149 +/- 0.012
```

**What it means:** All four conditions sit between 1.12 and 1.15 — close to 1.0, well below any threshold of concern (~2.0). Adding robustness constraints doesn't push people into implausible corners of feature space.

---

## Unit 5 — L2 Proximity

**Why we use it:** A standard metric from the recourse literature for measuring how far the recommended action moves someone in feature space. Reporting it makes results comparable to prior work.

**What we do:** Compute the Euclidean distance between the original and post-action feature vectors, over post-action observed features only. Still-missing features are excluded — their apparent "change" would just be the imputed mean minus zero, which reflects data representation rather than anything the person actually did.

**Results (4-fold CV):**

```
Condition     L2 Proximity
no-robust     0.912 +/- 0.053
baseline      1.026 +/- 0.035
ours          1.241 +/- 0.122
no-reveal     1.565 +/- 0.049
```

**What it means:** More constraints = further movement, as expected. no-reveal is highest because without reveals it must compensate entirely through edits. The standard errors confirm this ordering is stable across folds.

---

## Unit 6 — Retrain Robustness vs. Multiplicity Level

**Why we use it:** Units 2 and 3 evaluate robustness at a single operating point. But we need to show how robustness degrades as the set of plausible alternative models grows. This is the standard evaluation in the robust recourse literature (ElliCE Figure 1) — a curve of robustness vs. model multiplicity level.

**What we do:** Train 50 bootstrap models once per fold. For each multiplicity level epsilon_target, filter to only those bootstrap models whose Rashomon distance (1/2)(theta - theta_hat)^T H (theta - theta_hat) is within epsilon_target. Then check: for each person, does the recourse hold for ALL surviving models? Report the fraction of people for whom it does.

This uses the ElliCE-style robustness metric: binary per person (all-or-nothing), averaged across people. Stricter than the fraction-of-models metric in Unit 2.

**Results (4-fold CV):**

```
epsilon_target  #models   ours          baseline      no-robust     no-reveal
0.001           0.0       —             —             —             —
0.005           2.8       1.000+/-0.000 0.565+/-0.088 0.146+/-0.119 1.000+/-0.000
0.010           21.2      0.718+/-0.098 0.060+/-0.039 0.000+/-0.000 0.888+/-0.075
0.020           46.8      0.370+/-0.149 0.000+/-0.000 0.000+/-0.000 0.620+/-0.080
0.050           50.0      0.315+/-0.144 0.000+/-0.000 0.000+/-0.000 0.576+/-0.082
0.100           50.0      0.315+/-0.144 0.000+/-0.000 0.000+/-0.000 0.576+/-0.082
```

**What it means:** ours degrades gracefully — at epsilon_target=0.005 (only the ~3 most similar bootstrap models), 100% of people's recourse holds. baseline and no-robust collapse quickly. Note that epsilon_target=0.001 is empty because no bootstrap model happened to land that close to theta_hat (minimum distance was ~0.003).

---

## Unit 7 — Ellipsoid Robustness vs. Multiplicity Level

**Why we use it:** The bootstrap retrain evaluator in Unit 6 has a coverage problem — at tight epsilon_target, few or no bootstrap models qualify, leaving gaps in the curve. Ellipsoid sampling fixes this by drawing models directly from the Rashomon ellipsoid, giving exactly 50 models at every epsilon_target with uniform geometric coverage.

**What we do:** For each epsilon_target, sample 50 parameter vectors theta from the Rashomon ellipsoid {theta : (1/2)(theta - theta_hat)^T H (theta - theta_hat) <= epsilon_target} using the Cholesky decomposition of H. Check if the recourse holds for ALL 50 sampled models per person. Report the fraction of people for whom it does.

This evaluator probes the same geometric object the SOCP constraint optimizes against — the Hessian ellipsoid. It's a harder test than bootstrap retrain because it covers the region uniformly rather than relying on where bootstrap models happen to land.

**Results (4-fold CV):**

```
epsilon_target  ours          baseline      no-robust     no-reveal
0.001           1.000+/-0.000 0.490+/-0.119 0.008+/-0.007 1.000+/-0.000
0.005           0.873+/-0.040 0.032+/-0.019 0.008+/-0.007 0.972+/-0.024
0.010           0.546+/-0.126 0.000+/-0.000 0.000+/-0.000 0.787+/-0.112
0.020           0.245+/-0.127 0.000+/-0.000 0.000+/-0.000 0.458+/-0.140
0.050           0.036+/-0.031 0.000+/-0.000 0.000+/-0.000 0.074+/-0.040
0.100           0.000+/-0.000 0.000+/-0.000 0.000+/-0.000 0.000+/-0.000
```

**What it means:** No gaps — every row has data. At epsilon_target=0.001 (the optimization epsilon), ours holds at 100% as the theory predicts. baseline is already at 49% — without the Rashomon constraint, about half the people's recourse fails even among very similar models. As epsilon_target grows, everyone degrades, but ours consistently dominates. At epsilon_target=0.100, nobody survives — the set of plausible models is too diverse for any fixed recourse to satisfy all of them.

---

## Unit 8 — AWP Robustness vs. Multiplicity Level

**Why we use it:** Units 6 and 7 test model robustness empirically. This tests the theoretical guarantee — the closed-form lower bound from Proposition 2 — at varying levels of model multiplicity.

**What we do:** For each epsilon_target, evaluate the Proposition 2 lower bound with epsilon_eval = epsilon_target (rho_eval stays at the optimization-time value). If the bound is non-negative, the recourse is certified robust at that multiplicity level.

**Results (4-fold CV):**

```
epsilon_target  ours          baseline      no-robust     no-reveal
0.001           0.758+/-0.210 0.000+/-0.000 0.000+/-0.000 0.750+/-0.217
0.005           0.000+/-0.000 0.000+/-0.000 0.000+/-0.000 0.000+/-0.000
0.010           0.000+/-0.000 0.000+/-0.000 0.000+/-0.000 0.000+/-0.000
```

**What it means:** The AWP bound passes for ~76% of people at epsilon_target=0.001 (roughly matching the tuned optimization epsilon) and fails beyond. The bound is not 100% at epsilon_target=0.001 because one fold tuned to epsilon=0.0005, so its solutions were only certified up to that lower level. baseline and no-robust correctly fail everywhere — they were never optimized with Rashomon robustness.

This confirms the theory and implementation are consistent. The AWP bound is tight at the optimization epsilon and doesn't extend beyond it. The empirical evaluators (Units 6–7) show that practical robustness extends further than what the bound certifies.

---

## Unit 9 — The Full Comparison

**Why we use it:** Units 1–8 explain each metric in isolation. This puts everything together across all folds and all conditions.

**Results (4-fold stratified CV, tuned hyperparameters, mean +/- SE):**

| Condition | Feasible | Cost | Nominal | Retrain | AWP | LOF | L2 |
|---|---|---|---|---|---|---|---|
| **ours** | 0.983+/-0.009 | 1.573+/-0.044 | 1.000+/-0.000 | 0.942+/-0.015 | 1.000+/-0.000 | 1.117+/-0.012 | 1.241+/-0.122 |
| **baseline** | 1.000+/-0.000 | 1.094+/-0.050 | 1.000+/-0.000 | 0.705+/-0.028 | 1.000+/-0.000 | 1.141+/-0.019 | 1.026+/-0.035 |
| **no-robust** | 1.000+/-0.000 | 0.947+/-0.061 | 1.000+/-0.000 | 0.505+/-0.018 | 1.000+/-0.000 | 1.136+/-0.020 | 0.912+/-0.053 |
| **no-reveal** | 0.902+/-0.044 | 1.565+/-0.049 | 1.000+/-0.000 | 0.978+/-0.007 | 1.000+/-0.000 | 1.149+/-0.012 | 1.565+/-0.049 |

**What it means:**

**Does Rashomon robustness help?** Yes. ours vs baseline: retrain validity jumps from 0.705 to 0.942. A 24pp improvement from the tuned epsilon alone. The ellipsoid evaluator confirms this — at epsilon_target=0.001, ours holds at 100% while baseline is at 49%.

**Does imputation robustness matter independently?** Yes. baseline vs no-robust: retrain validity goes from 0.505 to 0.705 even without Rashomon protection. Both sources of uncertainty are pulling real weight.

**Are reveals necessary?** Yes, for some people. no-reveal fails for ~10% of individuals that ours handles. For those people, disclosing a missing feature is the only path — no edit-only recourse exists under the robustness constraints.

**Does robustness hurt plausibility?** No. LOF is close to 1.0 across all conditions (1.12–1.15). Robustness constraints don't push people into unrealistic corners of feature space.

**Is the tuning stable?** Yes. Three of four folds selected epsilon=0.001, one selected epsilon=0.0005. All selected rho_coverage=0.85. The results are consistent regardless of which epsilon was chosen — the tuning confirms rather than changes the story.

**How does robustness degrade with multiplicity?** The ellipsoid robustness curve (Unit 7) shows ours degrades gracefully — 100% at epsilon_target=0.001, 87% at 0.005, 55% at 0.01. Baseline and no-robust collapse immediately. This is the key plot for the paper.

---

## Per-Person Qualitative Output

For each individual, the system prints a feature-level table showing the original values, which features were missing, what the action does to each one (edit, reveal, or leave unchanged), and the post-action values in original unstandardized units.

```
--------------------------------------------------------------------------------
Feature                       Original  Miss  Reveal  Post-Action      Change
--------------------------------------------------------------------------------
Pregnancies                       3.00    no       -         3.00           -
Glucose                          85.00    no       -        102.34      +17.34
BloodPressure                    70.00    no       -         70.00           -
SkinThickness                      ---   yes       -          ---           -
Insulin                            ---   yes     yes        142.00    revealed
BMI                              28.50    no       -         28.50           -
DiabetesPedigreeFunction          0.35    no       -          0.35           -
Age                              24.00    no       -         24.00           -
--------------------------------------------------------------------------------
  cost             : total=1.2043  reveal=0.5000  edit=0.7043
  nominal validity : True
  retrain validity : 1.000
  awp (sanity)     : True  lb=0.0000
  lof plausibility : 1.0423  (1.0 = in-distribution)
```

Immutable features (Pregnancies, DiabetesPedigreeFunction, Age) always show `-` in the Change column — the solver cannot recommend changing them.
