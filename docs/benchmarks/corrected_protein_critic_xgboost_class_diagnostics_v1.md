# Corrected ProteinCritic XGBoost class-level diagnostics

This is a post-hoc decomposition of the aggregate comparison in
[`corrected_protein_critic_xgboost_v1.json`](corrected_protein_critic_xgboost_v1.json),
not a separate replication. The analyzer refits raw-sequence XGBoost using the
already-selected parameters from that report and verifies that its aggregate test
scores exactly match. It makes no test-based hyperparameter or model choices.
The full per-class precision, recall, F1, support, and confusion matrices are in
[`corrected_protein_critic_xgboost_class_diagnostics_v1.json`](corrected_protein_critic_xgboost_class_diagnostics_v1.json).

The per-class decomposition covers raw-feature XGBoost versus ProteinCritic for
PFAM and EC. XGBoost on frozen ProteinCritic embeddings remains available in the
aggregate report but was not refit for class-level breakdown; extracting its
training embeddings on this CPU-only environment was disproportionately slow.
Stability remains a regression task and is not included in the class breakdown.

## Findings

### PFAM

Raw-feature XGBoost had higher test recall than ProteinCritic in 25 of 43 PFAM
classes; the critic was higher in 11, with 7 ties. It had zero recall in 4 classes
versus 13 for the critic. This is consistent with the aggregate balanced-accuracy
and macro-F1 gap (0.4716 / 0.4733 for raw XGBoost versus 0.3018 / 0.2670 for the
critic). The embedding-XGBoost aggregate result was 0.3781 / 0.3666.

Examples of larger raw-XGBoost recall advantages include PF00571 (7 test
examples; 0.857 vs 0.143), PF00149 (13; 0.692 vs 0), and PF13581 (8; 0.625 vs
0). There are counterexamples: the critic did better on PF00892 (17; 0.765 vs
0.529) and PF04055 (5; 0.600 vs 0.400). These per-class figures are noisy: 27 of
43 families have fewer than 10 labelled test examples, and 39 have fewer than
20. Treat the examples as error-localization clues, not independent evidence of
family-specific superiority.

The largest raw-XGBoost off-diagonal confusion was PF00892 predicted as PF00528
(8 cases). Other repeated confusions included PF12802→PF01047, PF00672→PF02518,
and PF00583→PF00126 (6 each). The critic's largest confusion was PF00293→PF00583
(6 cases); PF02518→PF00672 was next (5).

### EC

The EC aggregate remains close: balanced accuracy / macro-F1 was 0.2113 / 0.2054
for raw XGBoost, 0.2380 / 0.2233 for ProteinCritic, and 0.2603 / 0.2627 for
XGBoost on critic embeddings. The class breakdown explains why raw XGBoost's
aggregate balanced accuracy is lower despite strong recall for EC class 2:
class-2 recall was 0.880 for raw XGBoost versus 0.730 for the critic, but the
critic did better on classes 1 (0.118 vs 0.039), 3 (0.388 vs 0.376), 5 (0.048 vs
0), and 7 (0.333 vs 0.083). Class 4 had zero recall for both models.

Raw XGBoost predicted class 2 for 396 of 517 labelled test proteins, compared
with 326 for the critic; in particular, it confused class 3 as class 2 in 103
cases (the critic did so in 89). Class 2 is the largest test class (200 examples),
so this is a useful diagnostic of prediction concentration, not a reason to
retune on test.

## Interpretation and next action

The clearest follow-up is a validation-only PFAM audit: inspect training-label
support, per-class validation recall, and the critic's error patterns before
choosing any critic head or loss ablation. Do not use these test-class results to
select a new model or hyperparameter. Since the test set has now been inspected
at class level, preserve it as the reported v2 result and use validation for
experimentation; a fresh external or independently frozen test set is needed for
a later confirmatory claim.

For EC, inspect the class-2 prediction concentration and rare-class recall on
validation before deciding whether any intervention is warranted. For stability,
the independent scaffold-diversity task remains the prerequisite: the existing
62 labelled test proteins belong to one cluster, so no class-level or
cluster-generalization claim follows from their regression scores.
