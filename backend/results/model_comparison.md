# Week 4 - Baseline vs Improved Model Comparison

Dataset: EngSAF (train 4733 / validation 338 / test 506). Labels: 0 = incorrect, 1 = partially correct, 2 = correct.

Metrics: Accuracy; Macro-F1 (treats the three classes equally); QWK = quadratic weighted kappa (agreement with the human grade, penalises big mistakes more); MAE = mean absolute error in label points (lower is better); Partial recall = share of 'partially correct' answers the model recognised.

## Test set

| Model | Accuracy | Macro-F1 | QWK | MAE | Partial recall |
|---|---|---|---|---|---|
| Baseline - TF-IDF + LR (Week 2) | 0.5020 | 0.4688 | 0.3194 | 0.6225 | 0.2408 |
| V1 - MiniLM + LR (Week 3) | 0.5000 | 0.4752 | 0.3326 | 0.5988 | 0.3141 |
| V2 - Hybrid features + LR (Week 4) | 0.5257 | 0.5005 | 0.3643 | 0.5711 | 0.3770 |

## Validation set

| Model | Accuracy | Macro-F1 | QWK | MAE | Partial recall |
|---|---|---|---|---|---|
| Baseline - TF-IDF + LR (Week 2) | 0.5059 | 0.4820 | 0.4014 | 0.5888 | 0.1875 |
| V1 - MiniLM + LR (Week 3) | 0.5059 | 0.4966 | 0.4452 | 0.5533 | 0.2422 |
| V2 - Hybrid features + LR (Week 4) | 0.5828 | 0.5766 | 0.4793 | 0.4941 | 0.3750 |

## Confusion matrices (test set)

Rows = true label, columns = predicted label (0 / 1 / 2).

**Baseline - TF-IDF + LR (Week 2)**

| true \ pred | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 48 | 18 | 47 |
| 1 | 26 | 46 | 119 |
| 2 | 16 | 26 | 160 |

**V1 - MiniLM + LR (Week 3)**

| true \ pred | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 41 | 26 | 46 |
| 1 | 21 | 60 | 110 |
| 2 | 4 | 46 | 152 |

**V2 - Hybrid features + LR (Week 4)**

| true \ pred | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 44 | 34 | 35 |
| 1 | 26 | 72 | 93 |
| 2 | 14 | 38 | 150 |

## Ablation (test set, same classifier settings as V2)

| Features used | Accuracy | Macro-F1 | QWK |
|---|---|---|---|
| MiniLM cosine only | 0.4941 | 0.4695 | 0.3329 |
| + TF-IDF cosine | 0.5000 | 0.4749 | 0.3286 |
| + word overlap + length features | 0.5316 | 0.5088 | 0.3793 |
| + key-point coverage (full V2) | 0.5257 | 0.5005 | 0.3643 |

## V2 feature weights for the 'correct' class

Features are standardised, so weights are comparable. Positive = pushes the prediction towards 'correct'.

| Feature | Weight |
|---|---|
| minilm_cosine | +0.511 |
| tfidf_cosine | -0.038 |
| keypoint_coverage | -0.058 |
| word_overlap | +0.439 |
| log_student_len | +0.292 |
| log_reference_len | -0.298 |
| length_ratio | -0.195 |

V2 hyper-parameters (chosen on validation macro-F1): C=1.0, class_weight=None
