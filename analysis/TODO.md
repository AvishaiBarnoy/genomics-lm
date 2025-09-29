# Analysis TODOs

## Mutation Maps
- [ ] Ingest `outputs/scores/*.tsv` into pandas
- [ ] Per-position summaries: `best_delta`, `n_better`
- [ ] Top-K mutants per position (+ synonymous flag)
- [ ] Heatmap (positions × codons) of ΔlogP
- [ ] Export `summary.csv`, `topk.csv`, and plots

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

