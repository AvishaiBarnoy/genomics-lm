import torch

from scripts.train_biophysics_fusion import build_one_hot_lookup, train_encoder
from src.codonlm.biophysics import (
    NucleotideEncoder,
    generate_shape_training_data,
    load_nucleotide_encoder_state,
)
from src.codonlm.model_tiny_gpt import TinyGPT


def test_biophysics_encoder_and_fusion():
    # 1. Test NucleotideEncoder shapes
    encoder = NucleotideEncoder(d_shape=3)
    encoder.eval()

    # Input has shape (B, 3L, 4), where L = 20, so 3L = 60
    bx = torch.zeros(4, 60, 4)
    pred_shapes = encoder(bx)
    assert pred_shapes.shape == (4, 20, 3)

    # 2. Test synthetic training data generation
    train_x, train_y = generate_shape_training_data(num_samples=10, seq_len_codons=15)
    assert train_x.shape == (10, 45, 4)
    assert train_y.shape == (10, 15, 3)

    # 3. Test lookup table mapping
    itos = ["ATG", "A", "<BOS_CDS>", "<PAD_CDS>"]
    lookup = build_one_hot_lookup(itos, device=torch.device("cpu"))
    assert lookup.shape == (4, 3, 4)

    # Codon 'ATG' should be fully encoded: A=idx 0, T=idx 3, G=idx 2
    assert lookup[0, 0, 0] == 1.0  # A
    assert lookup[0, 1, 3] == 1.0  # T
    assert lookup[0, 2, 2] == 1.0  # G

    # Single nucleotide 'A' should be encoded at index 0, followed by zeros
    assert lookup[1, 0, 0] == 1.0  # A
    assert (lookup[1, 1] == 0.0).all()
    assert (lookup[1, 2] == 0.0).all()

    # Special token should be all zeros
    assert (lookup[2] == 0.0).all()

    # 4. Test generator embedding injection
    generator = TinyGPT(
        vocab_size=len(itos),
        block_size=64,
        n_layer=1,
        n_head=1,
        n_embd=16,
        use_shape_guidance=True,
    )
    generator.eval()

    dummy_tokens = torch.randint(0, len(itos), (2, 10))
    one_hots = lookup[dummy_tokens]  # (2, 10, 3, 4)
    one_hots = one_hots.view(2, 30, 4)

    shapes = encoder(one_hots)  # (2, 10, 3)
    logits, _ = generator(dummy_tokens, shape_embeddings=shapes)
    assert logits.shape == (2, 10, 4)


def test_encoder_checkpoint_loader_accepts_raw_and_engine_payloads(tmp_path):
    state = NucleotideEncoder().state_dict()
    raw = tmp_path / "raw.pt"
    engine = tmp_path / "engine.pt"
    torch.save(state, raw)
    torch.save({"training_contract_version": 1, "task": {"model": state}}, engine)

    for path in (raw, engine):
        loaded = load_nucleotide_encoder_state(path)
        assert loaded.keys() == state.keys()
        for name, tensor in state.items():
            assert torch.equal(loaded[name], tensor)


def test_encoder_training_is_collision_safe(tmp_path):
    root = tmp_path / "runs"
    for _ in range(2):
        result = train_encoder(
            out_dir=root,
            epochs=1,
            batch_size=2,
            train_samples=4,
            validation_samples=2,
            sequence_codons=3,
            device_name="cpu",
        )
        assert result.status == "complete"

    assert (root / "shape-encoder" / "checkpoints" / "biophysics_encoder.pt").is_file()
    assert (
        root / "shape-encoder-r002" / "checkpoints" / "biophysics_encoder.pt"
    ).is_file()


def test_encoder_interrupted_resume_matches_uninterrupted(tmp_path):
    common = {
        "epochs": 1,
        "batch_size": 2,
        "train_samples": 6,
        "validation_samples": 2,
        "sequence_codons": 3,
        "device_name": "cpu",
        "seed": 23,
    }
    reference_root = tmp_path / "reference"
    resumed_root = tmp_path / "resumed"
    train_encoder(out_dir=reference_root, run_id="reference", **common)
    interrupted = train_encoder(
        out_dir=resumed_root,
        run_id="interrupted",
        max_time_minutes=0,
        **common,
    )
    assert interrupted.status == "interrupted"
    last = resumed_root / "interrupted" / "checkpoints" / "last.pt"
    resumed = train_encoder(
        out_dir=resumed_root,
        run_id="interrupted",
        resume=last,
        **common,
    )
    assert resumed.status == "complete"

    reference_state = load_nucleotide_encoder_state(
        reference_root / "reference" / "checkpoints" / "biophysics_encoder.pt"
    )
    resumed_state = load_nucleotide_encoder_state(
        resumed_root / "interrupted" / "checkpoints" / "biophysics_encoder.pt"
    )
    for name, tensor in reference_state.items():
        assert torch.equal(resumed_state[name], tensor)
