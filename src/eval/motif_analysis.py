import numpy as np
import pandas as pd

def calculate_pwm(sequences, vocab):
    """
    Calculate Position Weight Matrix (PWM) from a list of sequences.
    
    Args:
        sequences (list of lists): List of token sequences (same length).
        vocab (list): List of tokens corresponding to PWM rows.
        
    Returns:
        pd.DataFrame: PWM with tokens as index and positions as columns.
    """
    if not sequences:
        return pd.DataFrame()
        
    L = len(sequences[0])
    N = len(sequences)
    vocab_size = len(vocab)
    
    # Initialize counts
    counts = np.zeros((vocab_size, L))
    token_to_idx = {tok: i for i, tok in enumerate(vocab)}
    
    for seq in sequences:
        for pos, tok in enumerate(seq):
            if tok in token_to_idx:
                counts[token_to_idx[tok], pos] += 1
                
    # Normalize to probabilities
    pwm = counts / N
    return pd.DataFrame(pwm, index=vocab, columns=range(L))

def get_consensus(pwm_df):
    """
    Get the consensus sequence from a PWM DataFrame.
    """
    if pwm_df.empty:
        return ""
    return "".join(pwm_df.idxmax(axis=0).tolist())

def get_shannon_entropy(pwm_df):
    """
    Calculate Shannon Entropy per position.
    """
    if pwm_df.empty:
        return np.array([])
    # -sum(p * log2(p))
    # Avoid log(0)
    p = pwm_df.values
    entropy = -np.sum(p * np.log2(p + 1e-9), axis=0)
    return entropy
