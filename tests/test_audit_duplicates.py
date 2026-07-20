import numpy as np
import json
import subprocess
from pathlib import Path

def test_audit_duplicates_logic(tmp_path):
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    
    # vocab sizes and setup
    # 0: PAD, 4+: sense codons
    # Train sequences:
    # 1. [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] (len 12)
    # 2. [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30] (len 11)
    # 3. [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50] (len 11)
    X_train = np.zeros((3, 16), dtype=np.int64)
    X_train[0, :12] = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    X_train[1, :11] = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    X_train[2, :11] = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    
    # Test sequences:
    # 1. [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] (Exact duplicate of train seq 1)
    # 2. [90, 91, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 92] (Shares 10-mer [20-29] with train seq 2)
    # 3. [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110] (Completely unique)
    X_test = np.zeros((3, 16), dtype=np.int64)
    X_test[0, :12] = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    X_test[1, :13] = [90, 91, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 92]
    X_test[2, :11] = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    
    np.savez(train_npz, X=X_train)
    np.savez(test_npz, X=X_test)
    
    # Run the script
    cmd = [
        "python",
        "-m",
        "scripts.audit_duplicates",
        "--train_npz", str(train_npz),
        "--test_npz", str(test_npz)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed: {res.stderr}\nStdout: {res.stdout}"
    
    # Expected assertions:
    # 1 exact duplicate out of 3 -> 33.33%
    assert "Exact duplicate sequences in test split: 1 / 3 (33.33%)" in res.stdout
    # L=10: test seq 1 (exact) and test seq 2 (10-mer overlap) should both trigger overlap -> 2 out of 3 -> 66.67%
    assert "10 codons" in res.stdout
    assert "66.67%" in res.stdout
    # L=30: none should trigger overlap except maybe exact (which is len 12 < 30) so 0%
    assert "30 codons" in res.stdout
    assert "0.00%" in res.stdout

    report_path = tmp_path / "audit.json"
    blocked = subprocess.run(
        [
            *cmd,
            "--fail-on-exact",
            "--report-json",
            str(report_path),
        ],
        capture_output=True,
        text=True,
    )
    assert blocked.returncode == 4
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["exact_duplicates"] == 1
