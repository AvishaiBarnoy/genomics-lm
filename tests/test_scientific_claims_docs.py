from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_readme_does_not_present_legacy_stage26_metrics_as_current_headlines():
    readme = _read("README.md")

    assert "Stage 2.6 (current best)" not in readme
    assert "**Test PPL: 68.5**" not in readme
    assert "legacy protocol" in readme.lower()


def test_benchmark_table_declares_evidence_and_validation_axes():
    table = _read("conference/sota_benchmark_table.md")

    assert "Evidence source" in table
    assert "Validation status" in table
    assert "Legacy/leaky" in table
    assert "Controlled" in table
    assert "Independently replicated" in table


def test_active_scientific_docs_do_not_make_banned_unqualified_claims():
    paths = [
        "conference/sota_benchmark_table.md",
        "conference/project_story.md",
        "docs/DEVELOPMENT_LOG.md",
        "docs/RELEASE_NOTES.md",
    ]
    banned = [
        "orders of magnitude higher performance density",
        "publishable insight",
        "clinically significant finding",
        "hallmark of disentangled biological representations",
        "validating the data-scaling approach",
    ]

    combined = "\n".join(_read(path).lower() for path in paths)
    for claim in banned:
        assert claim not in combined
