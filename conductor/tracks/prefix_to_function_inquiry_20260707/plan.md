# Scientific Inquiry: Prefix-to-Function Experiment Plan

## Status
Planned.

## Objectives
1. **Context-to-Function Evaluation**: Measure how much biological context and functional class information is encoded in the first $k$ codons of a gene sequence.
2. **Dynamic Generation & Probing**: Feed the model varying prefix lengths $k$, generate complete open reading frames, and classify the continuations using pretrained downstream classifiers (e.g. EC level-1 classification).
3. **Information Saturation Analysis**: Analyze classification accuracy and agreement as a function of prefix length $k$ to locate where function info saturates.
4. **Reporting**: Compile charts and a markdown report summarizing the findings.

## Tasks
- [ ] Prepare a balanced test sequence set with diverse annotated functional classes (EC classes, AMR genes).
- [ ] Implement a prefix sweep loop that generates completions for $k \in \{1, 5, 10, 20, 30, 50, \dots\}$.
- [ ] Feed generated completions to ProteinCritic functional classification heads.
- [ ] Calculate semantic alignment metrics (true vs predicted class agreement/accuracy) for each prefix length.
- [ ] Generate prefix length vs. accuracy plots and compile a detailed markdown report under `docs/`.
- [ ] Write unit tests verifying prefix parsing and classification pipeline compatibility.
