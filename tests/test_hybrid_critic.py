import torch
import pytest
from src.codonlm.model_tiny_gpt import TinyGPT
from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.config import ProteinClassifierConfig
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.ebm import ProteinLatentEBM
from src.codonlm.generate import batch_score_critic, generate_cds_critic_guided

def test_batch_score_critic_and_guidance():
    # 1. Setup mock generator (TinyGPT)
    vocab_size = 69
    block_size = 32
    generator = TinyGPT(vocab_size=vocab_size, block_size=block_size, n_layer=1, n_head=1, n_embd=16)
    generator.eval()

    # Create dummy codon vocabularies
    codons = ["ATG", "AAA", "TGC", "TGG", "TTT", "CCA", "CCT", "CGT", "GTA", "GCT", "TAA", "TAG", "TGA", "<BOS_CDS>", "<EOS_CDS>", "<PAD_CDS>"]
    itos = codons + [f"C{i}" for i in range(vocab_size - len(codons))]
    stoi = {c: i for i, c in enumerate(itos)}

    # 2. Setup mock multitask critic
    p_tokenizer = ProteinTokenizer()
    critic_cfg = ProteinClassifierConfig(
        vocab_size=len(p_tokenizer),
        n_layer=1,
        n_head=1,
        n_embd=16,
        block_size=64,
        dropout=0.1,
        pooling="mean",
        num_classes=2,
    )
    task_dims = {"stability": 2, "function": 3}
    critic = MultiTaskProteinClassifier(critic_cfg, task_dims)
    critic.eval()

    # 3. Setup mock EBM
    ebm = ProteinLatentEBM(n_embd=16, hidden_dim=32)
    ebm.eval()

    # 4. Test batch scoring stability
    aa_seqs = ["MGEK", "MAPK"]
    scores_stab = batch_score_critic(
        critic_model=critic,
        tokenizer=p_tokenizer,
        aa_seqs=aa_seqs,
        target_task="stability",
        target_class_idx=0,
        device=torch.device("cpu")
    )
    assert scores_stab.shape == (2,)
    assert (scores_stab <= 0.0).all() # Log probability must be <= 0.0

    # 5. Test batch scoring EBM
    scores_ebm = batch_score_critic(
        critic_model=critic,
        tokenizer=p_tokenizer,
        aa_seqs=aa_seqs,
        target_task="ebm",
        target_class_idx=None,
        device=torch.device("cpu"),
        ebm_model=ebm
    )
    assert scores_ebm.shape == (2,)

    # 6. Test guided generation stability
    ctx_ids = [stoi["<BOS_CDS>"], stoi["ATG"]]
    gen_ids, info = generate_cds_critic_guided(
        model=generator,
        critic_model=critic,
        c_tokenizer=p_tokenizer,
        device=torch.device("cpu"),
        ctx_ids=ctx_ids,
        stoi=stoi,
        itos=itos,
        target_codons=10,
        hard_cap=20,
        alpha=0.5,
        guide_top_k=3,
        target_task="stability",
        target_class_idx=0,
        temperature=1.0,
        cds_only=True
    )
    assert len(gen_ids) > len(ctx_ids)
    assert info["generated_codons"] > 0

    # 7. Test guided generation EBM
    gen_ids_ebm, info_ebm = generate_cds_critic_guided(
        model=generator,
        critic_model=critic,
        c_tokenizer=p_tokenizer,
        device=torch.device("cpu"),
        ctx_ids=ctx_ids,
        stoi=stoi,
        itos=itos,
        target_codons=10,
        hard_cap=20,
        alpha=0.5,
        guide_top_k=3,
        target_task="ebm",
        target_class_idx=None,
        ebm_model=ebm,
        temperature=1.0,
        cds_only=True
    )
    assert len(gen_ids_ebm) > len(ctx_ids)
    assert info_ebm["generated_codons"] > 0
