# Deterministic check-up commands

- [x] Add tested CI, historical-run, and live-training inspection commands with shared Markdown/JSON reports.
- [x] Add explicit benchmark execution with recorded evidence and type-specific dispatch.
- [x] Verify focused tests, local CLI examples, and document usage and limits.

## Validation

- 26 focused tests; 90%+ coverage across the new modules.
- Repository-wide fatal-error Ruff check and diff whitespace check passed.
- Core suite: 482 passed, one skipped, one xpassed; two subprocess tests initially used the broken base SciPy environment, then both passed with the codonlm environment first on PATH.
- Live read-only CI check against PR 174: lint/core-tests passed, revision and individual check links recorded.
- Tiny CPU CodonLM train/resume fixture and explicit codon-test benchmark passed.
- Tiny ProteinCritic fixture and explicit critic-validation benchmark passed.
- Markdown and JSON command outputs inspected; default inspection preserves input files and modification times.

## Limits

- Legacy runs without locks/completion markers remain unknown rather than being guessed complete or failed.
- Current checks do not deserialize checkpoints; historical checks only use safe tensor-only loading.
- Benchmarks are explicit CPU evaluations, not an automatic full scientific benchmark suite.

## Report feedback follow-up (2026-10-02)

- [x] Extract legacy ProteinCritic progress and distinguish target reached from clean completion.
- [x] Render ages in human units and explain missing health, task dimensions, multilabel configuration and lineage.
- [x] Add atomic shared-engine session telemetry and consume terminal states in live inspection.
- [x] Verify training/resume behavior and regenerate an example analysis without overwriting the user's report_temp.md.

- [x] Recover legacy CodonLM epoch/step progress from completed checkpoints and test target-reached reporting.

Follow-up validation: 495 core tests passed, one skipped, one xpassed with the codonlm environment first on `PATH`. Focused inspection/engine tests: 45 passed before the additional legacy CodonLM test; 92% coverage of inspection/status modules and 98% of the status logger at that point. Updated example: /tmp/genomics-run-analysis-updated.md.

## PR review fixes (2026-10-02)

- [x] Require a clean completion marker when a session snapshot says `complete`; a stale pre-resume snapshot alone cannot close the run.
- [x] Warn and continue when saved `model_spec` metadata is not a mapping.
- [x] Select CodonLM test tokens from the explicit manifest and hash the selected artifact in the benchmark receipt.

Validation after these fixes: 498 core tests passed, one skipped, one xpassed; focused inspection tests: 33 passed. Ruff fatal-error rules and diff whitespace check passed.
