# Corrected Termination Replay Protocol

## Purpose

The head-only condition preserved next-token quality but did not improve raw
termination. Its auxiliary head also collapsed the three intermediate distance
classes. Generated-prefix replay tests whether explicit supervision on the states
immediately preceding real model failures can correct that representation and, via
joint backbone fine-tuning, improve unrestricted termination.

Replay remains an auxiliary-head intervention. It does not directly add stop-token
cross-entropy to the language-model logits. Any raw-decoding improvement must arise
through shared-backbone co-adaptation.

## Replay Construction

- Source checkpoint: `corrected-termination-head-seed1337/best.pt`.
- Prefix source: frozen **training** split only; test and validation records are not
  used for replay training.
- Source selection: 200 records, seed 1337, round-robin across 20 training genomes.
- Prefix: one codon.
- Decoder: unrestricted model sampling, temperature 1.0, top-k disabled.
- Limit: 300 new tokens, with no CDS mask or forced termination.
- Captured failures: 79 hard-cap prefixes.
- Replay artifact SHA-256:
  `adb92c0b940d155d84ea6580c420089c07ad327cec31d28d0852989d757e87d2`.

The final failed-prefix state is assigned class 0 because the desired next token is
a termination token. The preceding 30 states receive classes from the same locked
distance edges `[0,3,10,30]` as native training. This produces 2,449 sparse labels:

| Class | Labels |
| ---: | ---: |
| 0 | 79 |
| 1 | 237 |
| 2 | 553 |
| 3 | 1,580 |
| 4 | 0 |

Square-root inverse-frequency weights over replay classes 0-3 are used. Class 4 is
absent by construction and its configured weight is inert.

## Training Condition

- Initialize from the evaluated head-only checkpoint.
- One native-corpus epoch; unchanged frozen dataset and vocabulary.
- Native termination loss remains enabled at weight 0.1.
- One replay batch of four failures per 16-microbatch optimizer group.
- Replay loss multiplier 3.2, yielding group-average weight 0.2.
- Backbone learning rate `2e-6`; termination-head learning rate `5e-5`.
- No shape encoder, multi-offset prior, critic, decoder bias, or forced stop.

## Evaluation And Decision

Repeat the head-only frozen test and matched raw-generation protocol. Promotion
requires test NLL regression no greater than 2% relative to the corrected primary,
fewer raw hard caps across the two locked seeds, and no short-length collapse.
Syntax-constrained and decoder-biased outputs remain separately labelled inference
interventions.
