# Multi-Frame Overlapping Gene Modeling Implementation Plan

## Phase 1: Data Pipeline
- **Task 1.1: Overlapping Gene Dataset Parser**
  - Update GenBank extraction pipelines to identify overlapping gene coordinate spans (CDS records sharing nucleotide positions in different frames).
- **Task 1.2: Multi-Frame Packer**
  - Create a dataset packing format that aligns codon tokens in all three forward frames alongside their raw nucleotide context.

## Phase 2: Model Architecture
- **Task 2.1: Multi-Frame Position & Frame Embeddings**
  - Implement frame identifier embeddings (Frame 0, 1, 2) in [model_tiny_gpt.py](file:///Users/User/github/genomics-lm/src/codonlm/model_tiny_gpt.py) and sum them with positional/token embeddings.
- **Task 2.2: Multi-Head Prediction**
  - Update model output projections to predict token classes across the three frame outputs when modeling overlapping zones.

## Phase 3: Training & Tests
- **Task 3.1: train_multi_frame.py**
  - Build the dedicated training script for multi-frame prediction.
- **Task 3.2: Tests**
  - Write unit tests checking that overlapping zones generate valid frame-shifted predictions and verify that the model correctly predicts overlapping reading frames.
