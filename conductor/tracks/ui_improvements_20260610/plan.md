# Interactive UI Playgrounds & Live Monitor Upgrades Plan

## Phase 1: Live Monitoring & ReD Sampling Visualizer
- **Task 1.1: Live Training curves**
  - Implement a polling widget that scans `runs/` for active jobs, reads `curves.csv` and metadata, and renders live Loss and Learning Rate charts.
- **Task 1.2: ReD Interactive Playground**
  - Integrate `src/eval/red_sampler.py` (or generation function supporting ReD) into the Playground generation view.
  - Expose parameters (`max_resets`, `len_threshold`) and write a live log of stochastically reset trajectories.

## Phase 2: Biophysical & Structural Alignments
- **Task 2.1: Aligned DNA Shape Chart**
  - Implement a function in `src/eval/visualizer.py` that computes DNAshape properties on the fly for any generated DNA sequence.
  - Render a synced line-alignment chart in Streamlit showing physical parameters directly beneath each corresponding codon text.
- **Task 2.2: Live Attention Heatmaps**
  - Implement a head-attention weight extractor in `inference_playground.py`.
  - Render an interactive 2D heatmap matrix of token-to-token attention values using Plotly or Seaborn in Streamlit.

## Phase 3: Mutation Alignment Comparison
- **Task 3.1: Synonymous Alignment Viewer**
  - Build a visual comparison table highlighting position-wise differences between a wild-type seed and optimized generated variants.
  - Display green/red delta flags showing how individual synonymous swaps affected predicted stability probabilities.

## Phase 4: Testing & Verification
- **Task 4.1: Regression Testing**
  - Update `tests/test_web_dashboard.py` to ensure all new dynamic widget callbacks and data plotting wrappers load correctly.
