#!/usr/bin/env bash
# place data .gbff files in data/raw/

eval "$(conda shell.bash hook)"
conda activate codonlm

# 1) Extract with meta
python -m src.codonlm.extract_cds_from_genbank_v2 \
  --gbff data/raw/GCF_000005845.2_ASM584v2_genomic.gbff \
  --out_txt  data/processed/cds_dna.txt \
  --out_meta data/processed/cds_meta.tsv

# 2) Tokenize to codon IDs (same as before)
python -m src.codonlm.codon_tokenize \
  --inp data/processed/cds_dna.txt \
  --out_ids data/processed/codon_ids.txt

# 3) Build datasets with genome-aware train/val/test
python -m src.codonlm.build_dataset_v2 \
  --ids data/processed/codon_ids.txt \
  --group_meta data/processed/cds_meta.tsv \
  --block_size 256 --windows_per_seq 2

# 4) Train with MPS autocast, optional checkpointing, cosine LR
RUN_ID=${RUN_ID:-tiny-demo-v2}
CKPT_ROOT="outputs/checkpoints/${RUN_ID}"
SCORES_ROOT="outputs/scores/${RUN_ID}"
python -m src.codonlm.train_codon_lm_v2 --config configs/tiny_mps_v2.yaml --run_id "${RUN_ID}"


# 5) Evaluate on val and test sets
python -m src.codonlm.eval_perplexity \
  --ckpt "${CKPT_ROOT}/best.pt" \
  --val_npz data/processed/val_bs256.npz || true
python -m src.codonlm.eval_perplexity \
  --ckpt "${CKPT_ROOT}/best.pt" \
  --val_npz data/processed/test_bs256.npz || true

# 6) Score mutations for one CDS (write to scores dir)
head -n1 data/processed/cds_dna.txt > data/processed/one_cds.txt
mkdir -p "${SCORES_ROOT}"
conda run -n codonlm python -m src.codonlm.score_mutations \
  --ckpt "${CKPT_ROOT}/best.pt" \
  --dna data/processed/one_cds.txt \
  --out "${SCORES_ROOT}/one_cds.tsv"

# 7) Mine motifs (quick & dirty)
python -m src.codonlm.mine_motifs \
  --ckpt "${CKPT_ROOT}/best.pt" \
  --npz data/processed/train_bs256.npz --k 9 --clusters 100

# 8) Collect artifacts and organize run outputs under runs/<RUN_ID>
python -m scripts.collect_artifacts_yaml "${RUN_ID}" configs/tiny_mps_v2.yaml || true
