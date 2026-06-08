import torch
import torch.nn as nn
import numpy as np

class MotifExtractor:
    """
    Extracts hidden-state embeddings from a TinyGPT model using a sliding window.
    """
    def __init__(self, model, window_size, stride):
        self.model = model
        self.window_size = window_size
        self.stride = stride
        self._captured_states = []

    def _get_window_indices(self, seq_len):
        indices = []
        for start in range(0, seq_len - self.window_size + 1, self.stride):
            indices.append((start, start + self.window_size))
        return indices

    def _hook_fn(self, module, input, output):
        # Capture the hidden state (output of the Block)
        # output is (B, T, D)
        self._captured_states.append(output.detach())

    def extract(self, input_ids, layer_idx=-1, exclude_ids=None):
        """
        Extract pooled hidden states for sliding windows.
        
        Args:
            input_ids (torch.Tensor): Tensor of shape (B, T)
            layer_idx (int or list): Indices of layers to extract from.
            exclude_ids (list): List of token IDs. If any token in a window 
                                matches these, the window is skipped.
            
        Returns:
            Tuple[torch.Tensor, list]: (embeddings, kept_indices)
                                       embeddings: (N_kept, D_total)
                                       kept_indices: list of (seq_idx, window_idx)
        """
        self._captured_states = []
        n_layers = len(self.model.blocks)
        
        if isinstance(layer_idx, int):
            target_layers = [layer_idx % n_layers]
        else:
            target_layers = sorted([i % n_layers for i in layer_idx])
            
        hooks = []
        for i in target_layers:
            hooks.append(self.model.blocks[i].register_forward_hook(self._hook_fn))
            
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            self.model(input_ids)
            
        for h in hooks:
            h.remove()
            
        if not self._captured_states:
            raise RuntimeError("No hidden states captured. Check layer_idx.")

        all_states = torch.cat(self._captured_states, dim=-1) # (B, T, D_total)
        
        B, T, D = all_states.shape
        exclude_set = set(exclude_ids) if exclude_ids else set()
        
        window_embeddings = []
        kept_metadata = [] # (batch_idx, start, end)

        for b in range(B):
            indices = self._get_window_indices(T)
            for start, end in indices:
                # Check for excluded tokens in this window
                window_tokens = input_ids[b, start:end].tolist()
                if exclude_set.intersection(window_tokens):
                    continue
                
                # (window_size, D)
                win = all_states[b, start:end, :]
                # Mean pool
                pooled = win.mean(dim=0) # (D)
                window_embeddings.append(pooled)
                kept_metadata.append((b, start, end))
            
        if not window_embeddings:
            return torch.empty(0, D), []
            
        res = torch.stack(window_embeddings, dim=0) # (N_kept, D)
        return res, kept_metadata
