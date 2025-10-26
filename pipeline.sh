#!/usr/bin/env bash

echo "[deprecated] pipeline.sh is deprecated; please use pipeline_v2.sh" >&2

# activate environment
#conda env update -n codonlm -f env/conda-environment.yml

#conda run -n codonlm pip install --upgrade --force-reinstall "numpy<2"
eval "$(conda shell.bash hook)"
#conda install -c pytorch pytorch torchvision torchaudio numpy
#conda init
conda activate codonlm

RUN_ID=${RUN_ID:-tiny-demo}
CKPT_ROOT="outputs/checkpoints/${RUN_ID}"
SCORES_ROOT="outputs/scores/${RUN_ID}"

# place data .gbff files in data/raw/

# extract CDS
python -m src.codonlm.extract_cds_from_genbank \
  --gbff data/raw/GCF_000005845.2_ASM584v2_genomic.gbff \
  --out  data/processed/cds_dna.txt --min_len 90

# Tokenize to codon IDs
python -m src.codonlm.codon_tokenize \
  --inp data/processed/cds_dna.txt \
  --out_ids data/processed/codon_ids.txt

# Build dataset (M2 example)
python -m src.codonlm.build_dataset \
  --ids data/processed/codon_ids.txt \
  --block_size 256 --windows_per_seq 2

# Train (deprecated path) — prefer pipeline_v2.sh which uses the v2 trainer
python -m src.codonlm.train_codon_lm --config configs/tiny_mps.yaml --run_id "${RUN_ID}"

# Evalutate
python -m src.codonlm.eval_perplexity \
  --ckpt "${CKPT_ROOT}/best.pt" \
  --val_npz data/processed/val_bs256.npz

# Score mutations for one CDS
#python -m src.codonlm.score_mutations \
#  --ckpt "${CKPT_ROOT}/best.pt" \
#  --dna data/processed/cds_dna.txt 
head -n1 data/processed/cds_dna.txt > data/processed/one_cds.txt
mkdir -p "${SCORES_ROOT}"
conda run -n codonlm python -m src.codonlm.score_mutations \
  --ckpt "${CKPT_ROOT}/best.pt" \
  --dna data/processed/one_cds.txt \
  --out "${SCORES_ROOT}/one_cds.tsv"

# Mine motifs (quick & dirty)
python -m src.codonlm.mine_motifs \
  --ckpt "${CKPT_ROOT}/best.pt" \
  --npz data/processed/train_bs256.npz --k 9 --clusters 100
