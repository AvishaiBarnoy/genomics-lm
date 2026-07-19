from collections import Counter
import csv
import subprocess

import numpy as np

from src.codonlm.lossless_packing import chunk_record, pack_chunks, packed_arrays


def _transition_counts(sequences):
    return Counter(
        (tokens[index], tokens[index + 1])
        for tokens in sequences
        for index in range(len(tokens) - 1)
    )


def _record(tokens, source_id="source-1", split="train"):
    return {
        "tokens": tokens,
        "source_id": source_id,
        "source_line_idx": 0,
        "fragment_line_idx": 0,
        "fragment_index": 0,
        "fragment_codon_start": 10,
        "fragment_codon_end": 10 + len(tokens) - 2,
        "split": split,
    }


def test_exact_block_sequence_is_one_chunk():
    record = _record([1, 10, 11, 12, 13, 2])

    chunks = chunk_record(record, block_size=5)

    assert len(chunks) == 1
    assert list(chunks[0].tokens) == record["tokens"]
    assert chunks[0].token_start == 0
    assert chunks[0].token_end == 6
    assert chunks[0].continues_from_previous is False
    assert chunks[0].continues_to_next is False


def test_one_token_over_block_produces_lossless_overlap():
    record = _record([1, 10, 11, 12, 13, 14, 2])

    chunks = chunk_record(record, block_size=5)

    assert [list(chunk.tokens) for chunk in chunks] == [
        [1, 10, 11, 12, 13, 14],
        [14, 2],
    ]
    assert chunks[0].continues_to_next is True
    assert chunks[1].continues_from_previous is True
    assert sum(chunk.tokens.count(1) for chunk in chunks) == 1
    assert sum(chunk.tokens.count(2) for chunk in chunks) == 1
    assert _transition_counts([list(chunk.tokens) for chunk in chunks]) == _transition_counts(
        [record["tokens"]]
    )


def test_multi_chunk_metadata_preserves_coordinates_and_split():
    record = _record([1, 10, 11, 12, 13, 14, 15, 16, 17, 2], split="test")

    chunks = chunk_record(record, block_size=4)

    assert [(chunk.token_start, chunk.token_end) for chunk in chunks] == [
        (0, 5),
        (4, 9),
        (8, 10),
    ]
    assert [(chunk.codon_start, chunk.codon_end) for chunk in chunks] == [
        (10, 14),
        (13, 18),
        (17, 18),
    ]
    assert all(chunk.split == "test" for chunk in chunks)
    assert _transition_counts([list(chunk.tokens) for chunk in chunks]) == _transition_counts(
        [record["tokens"]]
    )


def test_multi_packing_exposes_gene_boundaries_and_continuations():
    first = chunk_record(_record([1, 10, 2], source_id="gene-a"), block_size=6)
    second = chunk_record(_record([1, 11, 2], source_id="gene-c"), block_size=6)
    long = chunk_record(
        _record([1, 20, 21, 22, 23, 24, 25, 2], source_id="gene-b"), block_size=6
    )

    windows = pack_chunks(
        [*first, *second, *long], block_size=6, mode="multi", sep_id=3
    )

    spans = [span for window in windows for span in window.spans]
    assert {span.source_id for span in spans} == {"gene-a", "gene-b", "gene-c"}
    continuation_spans = [span for span in spans if span.continues_from_previous]
    assert len(continuation_spans) == 1
    assert continuation_spans[0].window_token_start == 0
    assert continuation_spans[0].source_id == "gene-b"
    assert any(
        left.source_id != right.source_id
        for window in windows
        for left, right in zip(window.spans, window.spans[1:])
    )


def test_packed_arrays_align_provenance_with_model_inputs():
    chunks = chunk_record(_record([1, 10, 11, 12, 13, 14, 2]), block_size=5)
    windows = pack_chunks(chunks, block_size=5, mode="dynamic", sep_id=3)

    dynamic = packed_arrays(windows, block_size=5, mode="dynamic")
    fixed = packed_arrays(windows, block_size=5, mode="single")

    assert dynamic["lengths"].tolist() == [6, 2]
    assert len(dynamic["X"]) == len(dynamic["segment_ids"])
    assert fixed["X"].shape == fixed["Y"].shape == (2, 5)
    assert fixed["segment_ids"].shape == fixed["source_positions"].shape == (2, 5)
    assert fixed["source_positions"][1, :1].tolist() == [5]


def test_legacy_builder_uses_lossless_dynamic_chunks(tmp_path):
    ids_path = tmp_path / "ids.txt"
    meta_path = tmp_path / "meta.tsv"
    out_dir = tmp_path / "out"
    sequences = [
        [1, 10, 11, 12, 13, 14, 2],
        [1, 20, 21, 22, 23, 24, 2],
        [1, 30, 31, 32, 33, 34, 2],
    ]
    ids_path.write_text("\n".join(" ".join(map(str, seq)) for seq in sequences) + "\n")
    with open(meta_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["genome", "source_id"], delimiter="\t")
        writer.writeheader()
        for index in range(3):
            writer.writerow({"genome": f"genome-{index}", "source_id": f"gene-{index}"})

    result = subprocess.run(
        [
            "python",
            "-m",
            "src.codonlm.build_dataset",
            "--ids",
            str(ids_path),
            "--group_meta",
            str(meta_path),
            "--block_size",
            "5",
            "--pack_mode",
            "dynamic",
            "--out_dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    observed = []
    for split in ("train", "val", "test"):
        with np.load(out_dir / f"{split}_bs5.npz") as data:
            offsets = np.concatenate([[0], np.cumsum(data["lengths"])])
            observed.extend(
                data["X"][offsets[i] : offsets[i + 1]].tolist()
                for i in range(len(data["lengths"]))
            )
        assert (out_dir / f"{split}_packing.tsv").exists()
    assert _transition_counts(observed) == _transition_counts(sequences)
