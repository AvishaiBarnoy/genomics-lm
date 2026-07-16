#!/usr/bin/env python3
"""
scripts/eval_ppl_baselines.py — Perplexity and Cross-Entropy Baselines Evaluator

This script calculates baseline metrics (Uniform, Unigram, 1st-order Markov, 2nd-order Markov)
on a given test/val dataset to contextualize language model performance.

Usage:
  python -m scripts.eval_ppl_baselines --train_npz data/processed/train_bs256.npz --test_npz data/processed/test_bs256.npz --vocab_size 69
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from collections import defaultdict
import numpy as np

def fit_baselines(train_npz_path: Path, vocab_size: int, alpha: float = 0.01):
    print(f"[baselines] Fitting baselines on {train_npz_path}...")
    with np.load(train_npz_path) as data:
        X = data["X"]
        if "Y" in data:
            Y = data["Y"]
        else:
            Y = np.zeros_like(X)
            Y[:, :-1] = X[:, 1:]
            
    unigram_counts = np.zeros(vocab_size)
    bigram_counts = defaultdict(lambda: np.zeros(vocab_size))
    trigram_counts = defaultdict(lambda: np.zeros(vocab_size))
    
    N, T = Y.shape
    for i in range(N):
        for t in range(T):
            target = Y[i, t]
            if target == 0:  # Ignore PAD
                continue
            
            unigram_counts[target] += 1
            
            prev1 = X[i, t]
            bigram_counts[prev1][target] += 1
            
            if t > 0:
                prev2 = X[i, t-1]
                trigram_counts[(prev2, prev1)][target] += 1
            else:
                trigram_counts[(0, prev1)][target] += 1
                
    # Normalize with Laplace smoothing alpha
    unigram_probs = (unigram_counts + alpha) / (np.sum(unigram_counts[1:]) + alpha * (vocab_size - 1))
    unigram_probs[0] = 0.0
    
    bigram_probs = {}
    for prev, counts in bigram_counts.items():
        total = np.sum(counts[1:])
        probs = (counts + alpha) / (total + alpha * (vocab_size - 1))
        probs[0] = 0.0
        bigram_probs[prev] = probs
        
    trigram_probs = {}
    for context, counts in trigram_counts.items():
        total = np.sum(counts[1:])
        probs = (counts + alpha) / (total + alpha * (vocab_size - 1))
        probs[0] = 0.0
        trigram_probs[context] = probs
        
    return unigram_probs, bigram_probs, trigram_probs

def evaluate_baselines(test_npz_path: Path, unigram_probs, bigram_probs, trigram_probs, vocab_size: int, alpha: float = 0.01):
    print(f"[baselines] Evaluating baselines on {test_npz_path}...")
    with np.load(test_npz_path) as data:
        X = data["X"]
        if "Y" in data:
            Y = data["Y"]
        else:
            Y = np.zeros_like(X)
            Y[:, :-1] = X[:, 1:]
            
    total_tokens = 0
    uniform_nll = 0.0
    unigram_nll = 0.0
    bigram_nll = 0.0
    trigram_nll = 0.0
    
    default_probs = (np.zeros(vocab_size) + alpha) / (alpha * (vocab_size - 1))
    default_probs[0] = 0.0
    
    N, T = Y.shape
    for i in range(N):
        for t in range(T):
            target = Y[i, t]
            if target == 0:
                continue
                
            total_tokens += 1
            
            # Uniform over active classes (excluding PAD)
            uniform_nll += -math.log(1.0 / (vocab_size - 1))
            
            # Unigram
            unigram_nll += -math.log(max(1e-15, unigram_probs[target]))
            
            # Bigram (1st-order Markov)
            prev1 = X[i, t]
            if prev1 in bigram_probs:
                p_bigram = bigram_probs[prev1][target]
            else:
                p_bigram = default_probs[target]
            bigram_nll += -math.log(max(1e-15, p_bigram))
            
            # Trigram (2nd-order Markov)
            if t > 0:
                prev2 = X[i, t-1]
                context = (prev2, prev1)
            else:
                context = (0, prev1)
                
            if context in trigram_probs:
                p_trigram = trigram_probs[context][target]
            elif prev1 in bigram_probs:
                p_trigram = bigram_probs[prev1][target]
            else:
                p_trigram = default_probs[target]
            trigram_nll += -math.log(max(1e-15, p_trigram))
            
    results = {
        "Uniform": {
            "loss": uniform_nll / total_tokens,
            "ppl": math.exp(uniform_nll / total_tokens)
        },
        "Unigram": {
            "loss": unigram_nll / total_tokens,
            "ppl": math.exp(unigram_nll / total_tokens)
        },
        "Bigram (1st-order Markov)": {
            "loss": bigram_nll / total_tokens,
            "ppl": math.exp(bigram_nll / total_tokens)
        },
        "Trigram (2nd-order Markov)": {
            "loss": trigram_nll / total_tokens,
            "ppl": math.exp(trigram_nll / total_tokens)
        }
    }
    return results, total_tokens

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True, help="Path to train NPZ")
    ap.add_argument("--test_npz", required=True, help="Path to test NPZ")
    ap.add_argument("--vocab_size", type=int, default=69, help="Vocabulary size (excluding PAD)")
    ap.add_argument("--alpha", type=float, default=0.01, help="Laplace smoothing alpha")
    args = ap.parse_args()

    train_path = Path(args.train_npz)
    test_path = Path(args.test_npz)
    
    if not train_path.exists():
        raise FileNotFoundError(f"Train NPZ not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test NPZ not found: {test_path}")

    unigram_p, bigram_p, trigram_p = fit_baselines(train_path, args.vocab_size, args.alpha)
    results, tokens = evaluate_baselines(test_path, unigram_p, bigram_p, trigram_p, args.vocab_size, args.alpha)
    
    print("\n" + "=" * 55)
    print(f" Baseline Perplexity Comparison (Evaluated on {tokens} tokens)")
    print("=" * 55)
    print(f"{'Baseline Model':<30} | {'Cross-Entropy':<10} | {'Perplexity':<10}")
    print("-" * 55)
    for model_name, metrics in results.items():
        print(f"{model_name:<30} | {metrics['loss']:<10.4f} | {metrics['ppl']:<10.2f}")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()
