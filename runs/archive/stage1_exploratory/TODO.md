# Analysis TODOs

## Mutation Maps
- [x] Ingest `outputs/scores/<RUN_ID>/*.tsv` into pandas
- [x] Per-position summaries: `best_delta`, `n_better`
- [x] Top-K mutants per position (+ synonymous flag)
- [x] Heatmap (positions × codons) of ΔlogP
- [x] Export `summary.csv`, `topk.csv`, and plots

## Motif Mining
- [ ] Extract n-gram/k-mer motifs from high-ΔlogP regions
- [ ] Compare motif frequencies vs background
- [ ] Visualize motif logos

## Codon Language Model
- [ ] Track training curves (loss, ppl)
- [ ] Ablations: block_size, n_layer, n_head, n_embd
- [ ] Bias checks: AA-frozen shuffle vs original

## Bias & Validation
- [ ] Synonymous vs non-synonymous breakdown
- [ ] Cross-dataset evaluation (organism/taxonomy mix)
- [ ] Correlate ΔlogP with conservation scores (if available)
