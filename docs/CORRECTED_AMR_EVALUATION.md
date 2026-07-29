# Corrected AMR Evaluation

## Protocol

The corrected AMR benchmark uses six CARD drug classes with protein clusters held
out between 3,733 probe-training and 1,285 test sequences. Twenty-five exact
LM-training matches were removed before splitting, and the final audit reports
zero exact test/pretraining duplicates. Thirty-four protein clusters overlap LM
pretraining and are disclosed as an exposure covariate.

Both corrected CodonLM checkpoints use canonical causal final-layer hidden states,
mean pooled over non-PAD tokens. All embedding conditions use the same standardized
logistic-regression probe (`C=1`). Confidence intervals use 1,000 stratified
bootstrap resamples.

## Results

| Representation | Balanced accuracy | Macro-F1 | AUROC | Macro-AUPRC |
| --- | ---: | ---: | ---: | ---: |
| CodonLM seed 1337 | 0.322 | 0.317 | 0.777 | 0.312 |
| CodonLM seed 2027 | 0.349 | 0.347 | 0.795 | 0.331 |
| Random Transformer seed 19 | **0.508** | 0.422 | **0.898** | **0.526** |
| Random Transformer seed 23 | 0.503 | **0.444** | 0.879 | 0.474 |
| Nucleotide 3-mer TF-IDF | 0.194 | 0.187 | 0.837 | 0.342 |

Raw accuracy is approximately 0.78-0.81 across conditions but is not the primary
metric because 1,015 of 1,285 test records are beta-lactam examples.

## Interpretation

The pretrained checkpoints outperform the nucleotide 3-mer baseline on balanced
accuracy and macro-F1, but not consistently on AUROC or macro-AUPRC. More
importantly, both deterministic random-Transformer controls outperform both
pretrained checkpoints on every class-aware metric. The pretraining representation
gate therefore fails.

This does not contradict the intrinsic PPL result. Next-token pretraining improved
sequence prediction beyond trigram, but a final-layer causal mean is not guaranteed
to be the best gene-level representation. Random nonlinear features can preserve
global composition and length signals that the trained final layer suppresses while
specializing for next-token logits.

The next AMR work must be a predeclared representation ablation, not checkpoint
selection on AMR test results. Candidate extraction methods are:

1. mean pooling from earlier and middle Transformer layers;
2. EOS or final-codon pooling, which has access to the full causal prefix;
3. concatenated multi-layer pooling with a fixed dimensionality reduction;
4. centered pooling that excludes BOS/EOS and explicitly controls for sequence
   length and codon composition.

These methods should be selected using grouped cross-validation inside the AMR
training partition. The held-out AMR test set should not be reused for choosing the
pooling method.

## Train-Only Representation Selection

Five-fold stratified protein-cluster-grouped cross-validation was run on the 3,733
AMR training records only. It compared layers 0, 2, 5, 8, and final with
content-only mean, non-PAD mean, and EOS pooling for both CodonLM seeds. Macro-AUPRC
was the declared selection metric.

Layer 2 content-only mean pooling ranked first with mean macro-AUPRC `0.4587`
(fold/seed standard deviation `0.1109`). Layer 2 non-PAD mean was second at
`0.4576`; the small margin is a limitation. After locking the winner, it was
evaluated once on the held-out AMR test set:

| Representation | Balanced accuracy | Macro-F1 | AUROC | Macro-AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Layer 2 content mean, seed 1337 | **0.501** | **0.431** | **0.881** | 0.447 |
| Layer 2 content mean, seed 2027 | 0.469 | 0.405 | 0.875 | **0.451** |
| Final causal mean, seed 1337 | 0.322 | 0.317 | 0.777 | 0.312 |
| Final causal mean, seed 2027 | 0.349 | 0.347 | 0.795 | 0.331 |

Early-layer pooling repairs much of the final-layer deficit. It brings CodonLM
close to random-Transformer balanced accuracy (`0.503-0.508`) but still does not
consistently exceed the random controls, whose macro-AUPRC is `0.474-0.526`.
Therefore the extraction failure is confirmed, but an AMR-specific pretraining
advantage remains unproven.
