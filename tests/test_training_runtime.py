import pytest
import torch

from src.training.runtime import save_artifact_atomic


@pytest.mark.parametrize(
    "filename",
    [
        "last.pt",
        "best.pt",
        "last_critic.pt",
        "best_ebm.pt",
        "best-epoch-003.pt",
    ],
)
def test_artifact_writer_rejects_canonical_checkpoint_names(tmp_path, filename):
    with pytest.raises(ValueError, match="owned by TrainingEngine"):
        save_artifact_atomic({"weight": torch.tensor([1.0])}, tmp_path / filename)
    assert not (tmp_path / filename).exists()


def test_artifact_writer_accepts_descriptive_noncanonical_name(tmp_path):
    path = tmp_path / "biophysics_encoder.pt"
    payload = {"weight": torch.tensor([1.0])}
    save_artifact_atomic(payload, path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert torch.equal(saved["weight"], payload["weight"])
