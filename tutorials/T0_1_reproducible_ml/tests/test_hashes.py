import torch, hashlib
from repro.hashutil import tensor_checksum

def test_tensor_checksum_stable():
    p1 = torch.arange(10, dtype=torch.float32)
    p2 = p1.clone()
    assert tensor_checksum([p1]) == tensor_checksum([p2])
    p2[0] += 1e-6
    assert tensor_checksum([p1]) != tensor_checksum([p2])

