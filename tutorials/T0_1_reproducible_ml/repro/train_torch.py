import os, math, yaml, pathlib
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")  # skip torch.compile import path
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")    # belt & suspenders
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import random_split
from contextlib import nullcontext

from repro.seeds import set_global_seed
from repro.determinism import enable_full_determinism
from repro.hashutil import tensor_checksum, file_checksum
from repro.tiny_dataset import TinyLinearDataset, make_loader
from repro.tiny_model_torch import LinearTorch
from repro.logutil import get_logger

def device_select():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train(config_path="repro/config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    log = get_logger("torch", log_file="runs/log.txt")

    set_global_seed(cfg["seed"], mps=True)
    enable_full_determinism(raise_on_nondet=True)
    dev = device_select()
    log.info(f"Device: {dev}")

    ds = TinyLinearDataset(n=cfg["n"], d=cfg["d"], seed=cfg["data_seed"], noise_std=0.0)
    n_train = int(len(ds) * 0.8)
    n_val = len(ds) - n_train
    gsplit = torch.Generator().manual_seed(cfg["split_seed"])
    ds_train, ds_val = random_split(ds, [n_train, n_val], generator=gsplit)

    dl_train = make_loader(ds_train, cfg["batch_size"], seed=cfg["loader_seed"])
    dl_val   = make_loader(ds_val,   cfg["batch_size"], seed=cfg["loader_seed"])

    model = LinearTorch(d=cfg["d"]).to(dev)
    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # CUDA only; disabled on CPU/MPS

    # Autocast note: MPS supports float16 autocast for some ops, but for strict reproducibility keep it OFF here.
    autocast = nullcontext()

    accum = cfg["grad_accum_steps"]
    criterion = nn.MSELoss()

    # Save init checksum (to assert run-to-run identity)
    init_cksum = tensor_checksum(list(model.parameters()))
    log.info(f"init_checksum={init_cksum}")

    step = 0
    for ep in range(cfg["epochs"]):
        model.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)

        for i, (x, y) in enumerate(dl_train):
            x, y = x.to(dev), y.to(dev)

            with autocast:
                pred = model(x)
                loss = criterion(pred, y) / accum

            loss.backward()
            if (i + 1) % accum == 0:
                # Deterministic grad step
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1

            running += loss.item() * accum

        # validation
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for x, y in dl_val:
                x, y = x.to(dev), y.to(dev)
                pred = model(x)
                vloss += criterion(pred, y).item()
            vloss /= len(dl_val)

        log.info(f"epoch {ep:02d} | train_loss {running/len(dl_train):.8f} | val_loss {vloss:.8f}")

    # Save checkpoint and print checksum
    pathlib.Path("runs").mkdir(exist_ok=True, parents=True)
    ckpt = "runs/model.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt)
    model_cksum = tensor_checksum(list(model.parameters()))
    file_cksum = file_checksum(ckpt)
    log.info(f"final_param_checksum={model_cksum}")
    log.info(f"checkpoint_file_checksum={file_cksum}")
    return init_cksum, model_cksum, file_cksum

if __name__ == "__main__":
    train()

