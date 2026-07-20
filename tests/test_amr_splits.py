import csv
import json
import subprocess
from scripts.prepare_amr_dataset import _distribution_report
from tests.test_leakage_audit import _write_fake_mmseqs

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
        
        writer.writerow(["ARO:3000001", "beta-lactam antibiotic", "TEM beta-lactamase"])
        writer.writerow(["ARO:3000002", "beta-lactam antibiotic", "TEM beta-lactamase"])
        writer.writerow(["ARO:3000003", "beta-lactam antibiotic", "SHV beta-lactamase"])

    # 2. Create a mock FASTA file
    fasta_path = tmp_path / "card.fasta"
    sequences = {
        "a1": "ATG" + "GCT" * 20 + "TAA",
        "a2": "ATG" + "GAA" * 20 + "TAA",
        "b1": "ATG" + "TTT" * 20 + "TAA",
        "b2": "ATG" + "CCG" * 20 + "TAA",
    }
    with open(fasta_path, "w") as f:
        # aminoglycosides
        f.write(">gb|KP689347.1|ARO:3002523|AAC(2')-Ia\n" + sequences["a1"] + "\n")
        f.write(">gb|KP689348.1|ARO:3002524|AAC(2')-Ib\n" + sequences["a1"] + "\n")
        f.write(">gb|KP689349.1|ARO:3002525|AAC(3)-Ia\n" + sequences["a2"] + "\n")
        # beta-lactams
        f.write(">gb|TEM1|ARO:3000001|TEM-1\n" + sequences["b1"] + "\n")
        f.write(">gb|TEM2|ARO:3000002|TEM-2\n" + sequences["b1"] + "\n")
        f.write(">gb|SHV1|ARO:3000003|SHV-1\n" + sequences["b2"] + "\n")

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
    protocol_dir = out_dir / "annotation_family_held_out"
    train_csv = protocol_dir / "train_amr.csv"
    test_csv = protocol_dir / "test_amr.csv"
    assert train_csv.exists()
    assert test_csv.exists()

    # Sequence controls stay inside the explicit protocol output directory.
    train_seqs = protocol_dir / "train_amr_seqs.csv"
    test_seqs = protocol_dir / "test_amr_seqs.csv"
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

    report = json.loads((protocol_dir / "split_report.json").read_text())
    assert report["protocol"] == "annotation_family_held_out"
    assert "achieved_test_fraction" in report
    assert (protocol_dir / "split_assignments.tsv").exists()

    fake_mmseqs = tmp_path / "mmseqs"
    _write_fake_mmseqs(fake_mmseqs)
    cluster_cmd = cmd + [
        "--protocol", "protein_cluster_held_out",
        "--mmseqs-executable", str(fake_mmseqs),
    ]
    clustered = subprocess.run(cluster_cmd, capture_output=True, text=True)
    assert clustered.returncode == 0, clustered.stderr + clustered.stdout
    cluster_dir = out_dir / "protein_cluster_held_out"
    with (cluster_dir / "train_amr.csv").open() as handle:
        train_clusters = {row["protein_cluster"] for row in csv.DictReader(handle)}
    with (cluster_dir / "test_amr.csv").open() as handle:
        test_clusters = {row["protein_cluster"] for row in csv.DictReader(handle)}
    assert train_clusters.isdisjoint(test_clusters)
    cluster_report = json.loads((cluster_dir / "split_report.json").read_text())
    assert cluster_report["clustering"]["tool"]["version"] == "fake-mmseqs-1.0"
    assert cluster_report["train"]["records_per_class"]


def test_split_report_exposes_missing_classes():
    train = [{"drug_class": "a", "family": "fa", "protein_cluster": "ca"}]
    test = [{"drug_class": "b", "family": "fb", "protein_cluster": "cb"}]
    report = _distribution_report(train, test, "protein_cluster_held_out", 0.2)
    assert report["missing_train_classes"] == ["b"]
    assert report["missing_test_classes"] == ["a"]
