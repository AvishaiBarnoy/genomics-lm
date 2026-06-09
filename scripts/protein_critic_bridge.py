#!/usr/bin/env python3
"""
Stage 3: Protein-Critic Bridge
Translates CodonLM-generated DNA into Amino Acids and evaluates them using ProteinLM.
This hierarchical approach filters for functionally viable sequences.
"""

import torch
import torch.nn.functional as F
import argparse

from src.protein_lm.models import ProteinConditionalTransformer
from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.config import ProteinLMConfig, load_config

# Standard DNA translation table
CODON_TABLE = {
    "ATA": "I",
    "ATC": "I",
    "ATT": "I",
    "ATG": "M",
    "ACA": "T",
    "ACC": "T",
    "ACG": "T",
    "ACT": "T",
    "AAC": "N",
    "AAT": "N",
    "AAA": "K",
    "AAG": "K",
    "AGC": "S",
    "AGT": "S",
    "AGA": "R",
    "AGG": "R",
    "CTA": "L",
    "CTC": "L",
    "CTG": "L",
    "CTT": "L",
    "CCA": "P",
    "CCC": "P",
    "CCG": "P",
    "CCT": "P",
    "CAC": "H",
    "CAT": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGA": "R",
    "CGC": "R",
    "CGG": "R",
    "CGT": "R",
    "GTA": "V",
    "GTC": "V",
    "GTG": "V",
    "GTT": "V",
    "GCA": "A",
    "GCC": "A",
    "GCG": "A",
    "GCT": "A",
    "GAC": "D",
    "GAT": "D",
    "GAA": "E",
    "GAG": "E",
    "GGA": "G",
    "GGC": "G",
    "GGG": "G",
    "GGT": "G",
    "TCA": "S",
    "TCC": "S",
    "TCG": "S",
    "TCT": "S",
    "TTC": "F",
    "TTT": "F",
    "TTA": "L",
    "TTG": "L",
    "TAC": "Y",
    "TAT": "Y",
    "TAA": "_",
    "TAG": "_",
    "TGC": "C",
    "TGT": "C",
    "TGA": "_",
    "TGG": "W",
}


class ProteinCritic:
    def __init__(self, checkpoint_path, config_path, device="cpu"):
        self.device = torch.device(device)
        self.tokenizer = ProteinTokenizer()

        # Load Config
        self.config = load_config(config_path, ProteinLMConfig)
        self.config.vocab_size = len(self.tokenizer.vocab)

        # Build Model
        self.model = ProteinConditionalTransformer(self.config).to(self.device)

        # Load Weights
        print(f"[*] Loading ProteinLM from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        # Handle different checkpoint formats
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(sd)
        self.model.eval()

    def translate_dna(self, dna_seq):
        """Translates DNA string to Amino Acid string."""
        aa_seq = ""
        for i in range(0, len(dna_seq) - 2, 3):
            codon = dna_seq[i : i + 3].upper()
            aa = CODON_TABLE.get(codon, "X")
            if aa == "_":
                break  # Stop codon
            aa_seq += aa
        return aa_seq

    def score_sequence(self, aa_seq, conditions=None):
        """
        Computes the log-likelihood of the AA sequence under the ProteinLM.
        Higher score = more biologically 'natural' or 'likely' protein.
        """
        # Tokenize
        tokens = [self.tokenizer.bos_token_id]
        if conditions:
            tokens += self.tokenizer.encode_conditions(conditions)
        tokens += self.tokenizer.encode_sequence(aa_seq)
        tokens.append(self.tokenizer.eos_token_id)

        ids = torch.tensor([tokens], device=self.device).long()

        with torch.no_grad():
            # targets are shifted tokens
            targets = ids[:, 1:].contiguous()
            logits = self.model(ids[:, :-1]).contiguous()  # (1, T-1, V)

            # Cross entropy per position
            log_probs = F.log_softmax(logits, dim=-1)
            # Gather log probs of the actual target tokens
            target_log_probs = torch.gather(
                log_probs, -1, targets.unsqueeze(-1)
            ).squeeze(-1)

            total_log_prob = target_log_probs.sum().item()
            avg_log_prob = target_log_probs.mean().item()

        return {
            "total_log_prob": total_log_prob,
            "avg_log_prob": avg_log_prob,
            "perplexity": torch.exp(torch.tensor(-avg_log_prob)).item(),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dna", required=True, help="DNA sequence to evaluate")
    ap.add_argument(
        "--protein_ckpt", required=True, help="Path to ProteinLM checkpoint"
    )
    ap.add_argument("--protein_cfg", required=True, help="Path to ProteinLM config")
    ap.add_argument(
        "--conditions",
        nargs="*",
        help="Optional conditioning tokens, e.g. <FUNC:ENZYME>",
    )
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    critic = ProteinCritic(args.protein_ckpt, args.protein_cfg, device=device)

    aa_seq = critic.translate_dna(args.dna)
    print(f"[*] Translated Protein: {aa_seq}")

    scores = critic.score_sequence(aa_seq, conditions=args.conditions)

    print("\n=== Protein Critic Score ===")
    print(f"Total Log-Likelihood: {scores['total_log_prob']:.4f}")
    print(f"Avg Log-Likelihood per AA: {scores['avg_log_prob']:.4f}")
    print(f"Protein Perplexity: {scores['perplexity']:.4f}")
    print("============================")


if __name__ == "__main__":
    main()
