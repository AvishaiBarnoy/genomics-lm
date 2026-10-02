# Deterministic check-up scripts

Run these commands from the repository root in the project's Python environment.
All commands default to Markdown; `--format json` returns the same report data.
`--output PATH` saves the report instead of printing it. Inspection is read-only
unless an output path or benchmark is explicitly requested. Do not use an existing
training artifact as the output path: an explicit output file is overwritten.

## CI check-up

```bash
python -m scripts.check_ci --pr 123
python -m scripts.check_ci --pr 123 --format json --output /tmp/ci.json
```

Requires authenticated GitHub CLI (`gh`). Omitting `--pr` selects the current
branch's PR. `--repo OWNER/REPO` supports an explicit repository. Expected checks
default to `lint` and `core-tests`; repeat `--expect NAME` to replace that list.

Reports the current PR revision and its status-check rollup in one query, including
individual links and missing checks. Failed, cancelled, pending, unknown, skipped,
and neutral checks remain distinct; an empty rollup never passes. Review decision,
draft status, mergeability and merge state are separate from CI success. Expected
check names are not a claim about GitHub branch-protection requirements. This does
not watch, merge, approve, retry jobs, or modify the PR.

Exit codes: 0 = all checks passed and expected checks present; 1 = other CI state;
2 = collection error, including authentication, timeout, and unavailable `gh`.

## Current training check-up

```bash
python -m scripts.check_training --run-dir runs/YOUR_RUN
python -m scripts.check_training --run-dir runs/YOUR_RUN --quiet-minutes 15 --format json
```

Reads bounded log tails, saved metadata, curves, and checkpoint sizes/ages. It
probes the existing advisory lock without creating or truncating its file. It does
not load checkpoint tensors or attach to a process. The lock is stronger evidence
of run ownership than a PID that may have been reused.

Statuses:

| Status | Evidence |
| --- | --- |
| `active` | The run's advisory lock is held. This does not prove forward progress. |
| `active_quiet` | Lock held and no inspected log/curve/checkpoint file modified within the explicit quiet threshold. |
| `complete` | Valid completion marker and no observed active lock; a stale session snapshot alone cannot establish completion. |
| `failed` | Latest structured session records a catchable failure, with no held lock. |
| `interrupted` or `wall_time` | Latest structured session records an interrupted or configured wall-time stop, with no held lock. |
| `incomplete` | Released lock file, with no valid completion marker. |
| `unknown` | No conclusive lifecycle evidence, including legacy runs without lock files. |
| `error` | The run directory cannot be inspected. |

The report includes recent progress lines, parsed step/epoch/loss/LR telemetry when
available, the latest curve row, validation-loss change between the last two finite
rows, numerical-health evidence and error excerpts. Log messages can predate a
resume. Neither silence nor a traceback in an old log establishes a current crash.
Checkpoint and log modification times measure artifact activity, not optimizer
progress. There is no ETA extrapolation or automatic divergence judgment.

Exit codes: 0 = report produced (inspect its status); 2 = collection/input error.

## Old-run analysis

```bash
python -m scripts.analyze_run --run-dir runs/YOUR_RUN
python -m scripts.analyze_run --run-dir runs/YOUR_RUN \
  --output runs/YOUR_RUN/reports/analysis.md
```

Adds saved configuration, checkpoint progress/health, lineage, existing score JSON
files, and benchmark results/receipts. Chooses the newest `last` checkpoint for
progress, or the newest checkpoint when no `last` exists; this is not a model-quality
selection. Evaluation artifacts retain their own checkpoint provenance.

Auto-detects `codonlm` or `protein-critic` from saved metadata. Use
`--run-type codonlm` or `--run-type protein-critic` for legacy runs whose type is
unknown. Conflicting detection is reported rather than silently hidden. Current
checks may lack type evidence because they intentionally do not deserialize models.

Checkpoint inspection uses CPU and `torch.load(weights_only=True)`, with no unsafe
pickle fallback. Unsupported legacy checkpoints generate a warning; their other
artifacts can still be inspected. Existing score files are reported as recorded
results, not independently revalidated scientific claims. JSON files, curve files,
and each log tail have a 2 MB inspection bound; omissions/truncation are identified.

### Explicit benchmarks

Existing benchmark results are always included when present. Nothing executes by
default. These two suites require explicit input and checkpoint selection:

```bash
# CodonLM test perplexity/loss against a manifest-bound test set, on CPU.
python -m scripts.analyze_run --run-dir runs/YOUR_CODON_RUN \
  --benchmark codon-test \
  --checkpoint runs/YOUR_CODON_RUN/checkpoints/best.pt \
  --manifest data/processed/YOUR_DATASET/manifest.json \
  --output runs/YOUR_CODON_RUN/reports/analysis.md

# ProteinCritic validation metrics, on CPU; test split is not selected.
python -m scripts.analyze_run --run-dir runs/YOUR_CRITIC_RUN \
  --run-type protein-critic --benchmark critic-validation \
  --checkpoint runs/YOUR_CRITIC_RUN/checkpoints/best_critic.pt \
  --benchmark-config configs/YOUR_CRITIC_CONFIG.yaml \
  --benchmark-data data/processed/YOUR_DATASET/val.jsonl \
  --output runs/YOUR_CRITIC_RUN/reports/analysis.md
```

Each invocation writes a fresh directory under `reports/benchmarks/`, containing
`execution.json`, `benchmark.log`, and successful `result.json`. The receipt records
commands, Python version, CPU selection, timestamps, input SHA-256 hashes, timeout and exit status. Inputs are hashed again after execution; changed inputs invalidate a successful result.
CodonLM evaluation selects `test_tokens` from the supplied manifest and records
its hash in the receipt. It uses an isolated run layout; original score files are preserved.
Existing evaluators enforce their own dataset provenance rules. An active run lock
blocks benchmark execution. `--benchmark-timeout SECONDS` defaults to 3600.

A failed or timed-out benchmark remains in the report; it does not change the
historical training completion status. Exit codes: 0 = report produced and any
requested benchmark passed; 1 = benchmark failed/timed out; 2 = input/collection error.

## Reproducibility and limits

Reports use schema version 1, stable field/file ordering, strict JSON and explicit
UTC observation times. The same captured evidence produces the same classification;
observation times, artifact ages and live external state naturally change. Actual
benchmark measurements need not be bit-identical across environments.

Live filesystem inspection is a best-effort snapshot, not a transaction. Legacy
runs may lack provenance or reliable end-state records. Only score JSON files
immediately under `scores/` and this tool's benchmark receipts/results are collected;
plots, arbitrary CSV analyses and other output directories are not interpreted.

## Reading legacy reports and prediction heads

Markdown renders artifact ages in days/hours/minutes (for example `57d 1h`), while
JSON retains numerical seconds. Empty mappings explicitly say that the information
was not recorded. Missing telemetry is never interpreted as a zero error count.

Historical analysis reads both shared-engine and older ProteinCritic checkpoint
fields, including zero-based `epoch`, `epoch_complete`, `optimizer_step`,
`checkpoint_reason`, and dataset provenance. It also reads older CodonLM
`epoch`, `step`, and `epoch_microbatch_idx` progress fields. The completion-evidence section
separates reaching the configured epoch target from having a clean-completion
marker. A legacy run can have `target_reached: true` while its overall lifecycle
status remains `unknown`; the tool does not fabricate a completion marker.

Lineage records the source run/checkpoint of an explicit full-state fork. It is
not the loss history. No lineage artifact means ancestry is not recorded there;
it does not establish that the model was trained from scratch.

For ProteinCritic, task dimensions count output values, not dataset rows:

- `family`: Pfam protein-family classes.
- `function`: EC enzyme-function classes.
- `stability`: one continuous prediction when configured as a regression task;
  otherwise the number of stability classes.

Multi-task means predicting several properties (such as family and stability).
Multi-label means allowing several labels within a particular classification head.
An explicitly empty `multi_label_tasks` list disables multi-label heads. Its absence
is reported separately as not recorded.

## Structured logging for new training sessions

The shared engine now writes `run_status.json` atomically. The file is overwritten
at each session start with a new session ID; it describes the latest invocation,
not all past invocations. It contains start/update/finish times, launch mode,
model task/type, requested epochs, progress, latest metrics and their event context,
learning rates, committed units, aborted accumulation-group count, checkpoint name
and reason, and terminal status. Catchable failures include exception type/message.
The existing `run_complete.json` remains the clean-completion artifact.

Progress snapshots are throttled to one every 30 seconds. Phase boundaries,
checkpoint events and termination are written immediately. This is not a background
heartbeat: a long computation, force-kill or power loss can leave an old nonterminal
snapshot. The live checker uses the snapshot alongside the advisory lock and file
ages. A terminal snapshot with no held lock can establish `failed` or `interrupted`
(including a configured wall-time stop). Old runs are not retroactively given data.

Status logging errors warn without deliberately changing training control flow.
This telemetry records engine execution, starting at `fit()`; failures in earlier
CLI/config/dataset setup still require the existing logs. Detailed GPU telemetry
continues to come from trainer logs rather than this lightweight status snapshot.
