import torch
import numpy as np
import pytest
from src.codonlm.model_tiny_gpt import TinyGPT
from src.eval.motif_extractor import MotifExtractor

def test_sliding_window_indices():
    extractor = MotifExtractor(None, window_size=5, stride=2)
    seq_len = 10
    indices = extractor._get_window_indices(seq_len)
    # Expected windows: [0:5], [2:7], [4:9]
    # [6:11] is too far.
    assert indices == [(0, 5), (2, 7), (4, 9)]

def test_extraction_shape():
    vocab_size = 10
    block_size = 32
    n_embd = 64
    model = TinyGPT(vocab_size, block_size, n_layer=2, n_head=2, n_embd=n_embd)
    model.eval()
    
    window_size = 3
    stride = 1
    extractor = MotifExtractor(model, window_size=window_size, stride=stride)
    
    # 1 sequence of length 10
    input_ids = torch.randint(0, vocab_size, (1, 10))
    # windows: [0:3], [1:4], [2:5], [3:6], [4:7], [5:8], [6:9], [7:10] -> 8 windows
    
    embeddings, metadata = extractor.extract(input_ids, layer_idx=-1)
    # Expected shape: (N_windows, n_embd) = (8, 64)
    assert embeddings.shape == (8, n_embd)
    assert len(metadata) == 8

def test_multi_layer_extraction():
    vocab_size = 10
    block_size = 32
    n_embd = 64
    model = TinyGPT(vocab_size, block_size, n_layer=4, n_head=2, n_embd=n_embd)
    model.eval()
    
    extractor = MotifExtractor(model, window_size=3, stride=1)
    input_ids = torch.randint(0, vocab_size, (1, 10))
    
    # Extract from layers 1 and 3 (0-indexed)
    embeddings, metadata = extractor.extract(input_ids, layer_idx=[1, 3])
    # Expected shape: (8, 2 * 64) = (8, 128)
    assert embeddings.shape == (8, 128)

def test_exclusion_logic():
    vocab_size = 10
    block_size = 32
    n_embd = 64
    model = TinyGPT(vocab_size, block_size, n_layer=1, n_head=1, n_embd=n_embd)
    model.eval()
    
    extractor = MotifExtractor(model, window_size=3, stride=1)
    # Sequence with a 9 at index 5
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 9, 6, 7, 8]]) # len 9
    # Windows (WS=3): [0:3], [1:4], [2:5], [3:6], [4:7], [5:8], [6:9] -> 7 windows
    # Window [3:6] contains '9'
    # Window [4:7] contains '9'
    # Window [5:8] contains '9'
    # Windows kept: [0:3], [1:4], [2:5], [6:9] -> 4 windows
    
    embeddings, metadata = extractor.extract(input_ids, layer_idx=0, exclude_ids=[9])
    assert embeddings.shape[0] == 4
    kept_starts = [m[1] for m in metadata]
    assert kept_starts == [0, 1, 2, 6]
