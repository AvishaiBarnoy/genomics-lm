import hashlib
import torch

def tensor_checksum(params, algo="sha256"):
    h = hashlib.new(algo)
    with torch.no_grad():
        for p in params:
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def file_checksum(path, algo="sha256"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

