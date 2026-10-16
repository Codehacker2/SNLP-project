# Week 6 prep - V3 accuracy-improvement pass

Same EngSAF splits as Week 2-5 (train 4733 / validation 338 / test 506), untouched. V3 keeps every Week 4 V2 feature except `keypoint_coverage` (replaced by `keypoint_min_coverage`) and adds `negation_mismatch`. See `features.py` for why.

## Classifier search (validation macro-F1)

| Candidate | Validation macro-F1 |
|---|---|
| LogisticRegression(C=0.1, class_weight=None) | 0.5619 |
| LogisticRegression(C=0.1, class_weight=balanced) | 0.5592 |
| LogisticRegression(C=1.0, class_weight=None) | 0.5670 |
| LogisticRegression(C=1.0, class_weight=balanced) | 0.5592 |
| LogisticRegression(C=10.0, class_weight=None) | 0.5670 |
| LogisticRegression(C=10.0, class_weight=balanced) | 0.5570 |
| RandomForest(n_estimators=200, max_depth=None) | 0.5466 |
| RandomForest(n_estimators=200, max_depth=8) | 0.5274 |
| RandomForest(n_estimators=400, max_depth=None) | 0.5390 |
| RandomForest(n_estimators=400, max_depth=8) | 0.5247 |
| GradientBoosting(n_estimators=150, lr=0.05) | 0.5530 |
| GradientBoosting(n_estimators=150, lr=0.1) | 0.5463 |
| GradientBoosting(n_estimators=300, lr=0.05) | 0.5424 |
| GradientBoosting(n_estimators=300, lr=0.1) | 0.5302 |

**Selected: LogisticRegression(C=1.0, class_weight=None)**

## Test set: V2 (Week 4) vs V3 (this pass)

| Model | Accuracy | Macro-F1 | QWK | MAE | Partial recall |
|---|---|---|---|---|---|
| V2 - Hybrid features + LR (Week 4) | 0.5257 | 0.5005 | 0.3643 | 0.5711 | 0.3770 |
| V3 - LogisticRegression(C=1.0, class_weight=None) | 0.5494 | 0.5252 | 0.3919 | 0.5415 | 0.4084 |

## Validation set (V3)

| Metric | Value |
|---|---|
| accuracy | 0.5769 |
| macro_f1 | 0.5670 |
| qwk | 0.4788 |
| mae | 0.4941 |
| partial_recall | 0.3828 |
