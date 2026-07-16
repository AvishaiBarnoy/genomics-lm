import csv
import json
import subprocess
from pathlib import Path
import yaml

def test_homology_aware_amr_splits(tmp_path):
    # 1. Create a mock ARO index file
    aro_idx_path = tmp_path / "aro_index.tsv"
    with open(aro_idx_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        # Header
        writer.writerow(["ARO Accession", "Drug Class", "AMR Gene Family"])
        # Mock records: 3 in Class A (2 families), 3 in Class B (2 families)
        writer.writerow(["ARO:3002523", "aminoglycoside antibiotic", "AAC(2')"])
        writer.writerow(["ARO:3002524", "aminoglycoside antibiotic", "AAC(2')"])
        writer.writerow(["ARO:3002525", "aminoglycoside antibiotic", "AAC(3)"])
        
        writer.writerow(["ARO:3000001", "beta-lactam", "TEM beta-lactamase"])
        writer.writerow(["ARO:3000002", "beta-lactam", "TEM beta-lactamase"])
        writer.writerow(["ARO:3000003", "beta-lactam", "SHV beta-lactamase"])

    # 2. Create a mock FASTA file
    fasta_path = tmp_path / "card.fasta"
    mock_seq = "ATGACGATCACATTTTCGCGCCGGCAGGCGATTGCCGGCGCTCTCCTTGCCGTTCCCGCCGTGTCCACGCTGGCCGCC" # 26 codons
    with open(fasta_path, "w") as f:
        # aminoglycosides
        f.write(">gb|KP689347.1|ARO:3002523|AAC(2')-Ia\n" + mock_seq + "\n")
        f.write(">gb|KP689348.1|ARO:3002524|AAC(2')-Ib\n" + mock_seq + "\n")
        f.write(">gb|KP689349.1|ARO:3002525|AAC(3)-Ia\n" + mock_seq + "\n")
        # beta-lactams
        f.write(">gb|TEM1|ARO:3000001|TEM-1\n" + mock_seq + "\n")
        f.write(">gb|TEM2|ARO:3000002|TEM-2\n" + mock_seq + "\n")
        f.write(">gb|SHV1|ARO:3000003|SHV-1\n" + mock_seq + "\n")

    out_dir = tmp_path / "labels"
    out_dir.mkdir(exist_ok=True)

    # 3. Run prepare_amr_dataset script
    cmd = [
        "python",
        "-m",
        "scripts.prepare_amr_dataset",
        "--fasta", str(fasta_path),
        "--aro_index", str(aro_idx_path),
        "--out_dir", str(out_dir),
        "--min_examples", "2",
        "--top_n_classes", "2",
        "--test_frac", "0.33",
        "--seed", "1337"
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed: {res.stderr}\nStdout: {res.stdout}"

    # 4. Verify outputs exist
    train_csv = out_dir / "train_amr.csv"
    test_csv = out_dir / "test_amr.csv"
    assert train_csv.exists()
    assert test_csv.exists()

    # Also verify sequence CSVs are written to data/processed
    train_seqs = Path("data/processed/train_amr_seqs.csv")
    test_seqs = Path("data/processed/test_amr_seqs.csv")
    assert train_seqs.exists()
    assert test_seqs.exists()

    # 5. Read split records and check families
    train_families = set()
    with open(train_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_families.add(row["family"])

    test_families = set()
    with open(test_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_families.add(row["family"])

    print(f"Train families: {train_families}")
    print(f"Test families: {test_families}")

    # Check zero overlap between train and test families (Issue #54 success criteria)
    assert train_families.isdisjoint(test_families), "Overlapping families between train and test splits!"
