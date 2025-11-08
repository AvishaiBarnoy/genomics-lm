import numpy as np
import torch

from src.codonlm.generate import generate_cds_constrained, STOP_CODONS


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int, stop_id: int, prefer_id: int, stop_after: int, block_size: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.stop_id = stop_id
        self.prefer_id = prefer_id
        self.stop_after = int(stop_after)
        self.block_size = block_size
        self.calls = 0

    def forward(self, x, y=None):
        # x: (1, T)
        self.calls += 1
        T = x.shape[1]
        logits = torch.zeros((1, T, self.vocab_size), dtype=torch.float32, device=x.device)
        # By default prefer prefer_id; after stop_after codons, prefer stop_id
        # We can't easily count codons here; tests keep it aligned so call count approximates steps
        token = self.stop_id if self.calls >= self.stop_after else self.prefer_id
        logits[0, -1, token] = 10.0
        return logits, None


def test_constrained_generation_long_protein():
    # Build tiny vocab: 0:<PAD>,1:<BOS_CDS>,2:<EOS_CDS>,3:<SEP>,4:"AAA",5:"TAA"
    itos = ["<PAD>", "<BOS_CDS>", "<EOS_CDS>", "<SEP>", "AAA", "TAA"]
    stoi = {t: i for i, t in enumerate(itos)}
    prefer_id = stoi["AAA"]; stop_id = stoi["TAA"]
    model = DummyModel(vocab_size=len(itos), stop_id=stop_id, prefer_id=prefer_id, stop_after=360, block_size=512)
    device = torch.device("cpu")
    # ctx_ids: BOS + 20-codon prefix (simulate k=20)
    ctx_ids = [stoi["<BOS_CDS>"]] + [prefer_id] * 20
    target = 350
    hard_cap = 512 - 20 - 6  # block_size - k - margin
    ids, info = generate_cds_constrained(
        model=model,
        device=device,
        ctx_ids=ctx_ids,
        stoi=stoi,
        itos=itos,
        target_codons=target,
        hard_cap=hard_cap,
        require_terminal_stop=True,
        temperature=1.0,
        topk=0,
    )
    # Count generated codons after prefix
    codons = [itos[i] for i in ids if len(itos[i]) == 3 and set(itos[i]) <= set("ACGT")]
    gen_len = max(0, len(codons) - 20)
    assert gen_len >= 100  # default min_aa_len
    assert gen_len <= hard_cap
    # require terminal stop means either last is stop or we hit hard cap
    last_codon = codons[-1] if codons else ""
    assert (last_codon in STOP_CODONS) or bool(info.get("hit_hard_cap"))

