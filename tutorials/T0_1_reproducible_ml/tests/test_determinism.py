import subprocess, sys, os, shutil, hashlib, time

def test_two_runs_identical(tmp_path):
    # Run training twice and compare final checkpoints
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    for i in range(2):
        subprocess.run([sys.executable, "-m", "repro.train_torch"], check=True, env=env)
        shutil.copyfile("runs/model.pt", tmp_path / f"model_{i}.pt")
    # Compare file bytes
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f: h.update(f.read())
        return h.hexdigest()
    assert sha(tmp_path/"model_0.pt") == sha(tmp_path/"model_1.pt")

