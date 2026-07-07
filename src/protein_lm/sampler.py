import torch
import torch.nn as nn
import torch.nn.functional as F

from src.protein_lm.tokenizer import ProteinTokenizer

def latent_langevin_sample(
    ebm_model: nn.Module,
    critic_model: nn.Module,
    tokenizer: ProteinTokenizer,
    initial_seq: str,
    steps: int = 50,
    lr: float = 0.05,
    noise_std: float = 0.01,
    device: torch.device = torch.device("cpu")
) -> tuple[str, list[float]]:
    """
    Performs continuous Langevin dynamics optimization in the token-level embedding space 
    of ProteinLM to minimize the sequence energy.

    z_{t+1} = z_t - lr * \nabla_z E(z_t) + sqrt(2 * lr) * epsilon

    Args:
        ebm_model: The trained ProteinLatentEBM instance.
        critic_model: The frozen MultiTaskProteinClassifier instance containing the transformer backbone.
        tokenizer: The ProteinTokenizer instance.
        initial_seq: Initial protein sequence string.
        steps: Number of MCMC gradient steps.
        lr: Langevin dynamics step size (learning rate).
        noise_std: Noise multiplier coefficient.

    Returns:
        (optimized_sequence_str, energy_history)
    """
    critic_model.eval()
    ebm_model.eval()

    # 1. Tokenize initial sequence
    tokens = [tokenizer.bos_token_id] + tokenizer.encode_sequence(initial_seq) + [tokenizer.eos_token_id]
    tokens_t = torch.tensor([tokens], dtype=torch.long, device=device)

    # 2. Extract starting token embeddings (z) from backbone
    with torch.no_grad():
        token_embeddings_matrix = critic_model.backbone.token_embedding.weight.clone() # (vocab_size, n_embd)
        
        # We start optimization on the continuous token embeddings of our input sequence
        # shape: (1, seq_len, n_embd)
        z_start = critic_model.backbone.token_embedding(tokens_t)

    # Make z a leaf tensor requiring gradients
    z = z_start.clone().detach().requires_grad_(True)
    energy_history = []

    # 3. Langevin MCMC Optimization Loop
    for step in range(steps):
        # We simulate the rest of the classifier backbone forward pass on z
        seq_length = z.size(1)
        pos_embeds = critic_model.backbone.position_embedding(torch.arange(seq_length, device=z.device))
        x = critic_model.backbone.dropout(z + pos_embeds)

        is_causal = not getattr(critic_model.config, "bidirectional", True)
        if is_causal:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=z.device)
        else:
            causal_mask = None

        for block in critic_model.backbone.transformer_blocks:
            x = block(
                x,
                src_mask=causal_mask,
            )

        # Pool features
        if critic_model.pooling_type == "attention":
            pooled, _ = critic_model.pooler(x)
        else:
            pooled = x.mean(dim=1)

        # Bottleneck projection
        latent = critic_model.shared_latent(pooled)

        # Compute energy
        energy = ebm_model(latent)
        energy_val = energy.item()
        energy_history.append(energy_val)

        # Compute gradient w.r.t z
        grad = torch.autograd.grad(energy.sum(), z)[0]

        # Update z with Langevin step
        with torch.no_grad():
            noise = torch.randn_like(z) * noise_std
            z -= lr * grad + noise

    # 4. Project optimized embeddings back to discrete vocabulary space
    # Find nearest token index for each residue position (excluding BOS/EOS)
    optimized_tokens = []
    
    with torch.no_grad():
        for pos in range(1, z.size(1) - 1): # skip BOS (0) and EOS (len-1)
            z_pos = z[0, pos] # (n_embd,)
            # Compute Euclidean distances to all amino acids in the vocabulary
            distances = torch.norm(token_embeddings_matrix - z_pos, dim=1)
            
            # Restrict lookup to valid amino acids to prevent decoding into special condition/padding tokens
            aa_indices = torch.tensor([tokenizer.token_to_id[aa] for aa in tokenizer.amino_acids], dtype=torch.long, device=device)
            best_idx = aa_indices[torch.argmin(distances[aa_indices])].item()
            optimized_tokens.append(best_idx)

    optimized_seq = tokenizer.decode_sequence(optimized_tokens)
    return optimized_seq, energy_history
