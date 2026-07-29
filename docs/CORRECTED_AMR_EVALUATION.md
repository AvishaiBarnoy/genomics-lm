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
