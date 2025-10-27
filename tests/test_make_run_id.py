import datetime as _dt
from pathlib import Path

from scripts.make_run_id import make_run_id


def test_make_run_id_format(tmp_path, monkeypatch):
    cfg = {
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 128,
        "epochs": 5,
    }
    cfg_path = tmp_path / "tiny_mps.yaml"
    cfg_path.write_text("n_layer: 2\nn_head: 4\nn_embd: 128\nepochs: 5\n")

    class FakeDate(_dt.date):
        @classmethod
        def today(cls):
            return cls(2025, 10, 6)

    import scripts.make_run_id as mri
    monkeypatch.setattr(mri.dt, "date", FakeDate, raising=True)

    rid = make_run_id(cfg_path, cfg)
    assert rid == "2025-10-06_tiny_2L4H_d128_e5"
