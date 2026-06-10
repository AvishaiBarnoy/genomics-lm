# hayaData 2026 Fit Assessment

## Short Answer

Yes. Genomics-LM can fit hayaData 2026, especially if it is framed as a practical data and AI engineering case study rather than as a biology-only research talk.

hayaData is aimed at the data engineering, data science, BI, and analytics communities. The project has several natural entry points for that audience: custom biological data pipelines, compact transformer training, model analysis, generator-critic evaluation, and practical constraints around running ML on local hardware.

Conference reference: https://www.haya-data.com/

## Best Submission Angle

The strongest proposal is:

**GPT on a Shoestring Budget: Training Biological Language Models on Consumer Hardware**

This angle is broad enough for a data conference audience and concrete enough to avoid sounding like a niche genomics talk. The reusable lesson is not "biology is interesting"; it is how to build a full ML system under real constraints:

- Raw GenBank-style biological records into model-ready training data.
- Codon-level tokenization as a domain-specific compression strategy.
- Lightweight GPT-style training on limited hardware.
- Evaluation through generation behavior, biological validity checks, and critic models.
- Practical lessons from failures such as poor stop-codon termination.

## Recommended Categories

### Data Science / AI / Machine Learning

This is the best default category.

Use this if the proposal emphasizes CodonLM, ProteinLM, representation probing, generation quality, Reset-and-Discard sampling, or the generator-critic loop.

Possible title:

**Building a Small Genomic Language Model: What a Codon-Level GPT Learns from DNA**

### Data Engineering / Big Data Pipelines

This is a strong second option.

Use this if the proposal emphasizes parsing, cleaning, taxonomy handling, CDS extraction, tokenization, genomic tapes, and dataset construction. This version is likely easier for a broad data engineering audience to follow.

Possible title:

**From GenBank to Genomic GPT: Building Data Pipelines for Biological Language Models**

### MLOps / Practical ML / ML Engineering

Use this if the CFP has a practical ML, MLOps, or production ML category. If not, submit this version under Data Science / AI.

This angle should focus on hardware limits, memory-aware model design, reproducible runs, dashboard/demo workflows, cached examples, and benchmark reporting.

Possible title:

**GPT on a Shoestring Budget: Training Biological Language Models on Consumer Hardware**

### Applied Data Science / Other

Use this only if the available Sessionize categories are broad and none of the above match directly. The project is applied AI for life sciences, but the submission should still foreground data and ML engineering.

## What To Avoid

Avoid pitching the talk as a claim that the project is already comparable to frontier genomic foundation models unless the benchmark evidence is ready.

Avoid making "learns 3D DNA physics" the central claim unless the structural probing results, plots, and baselines are polished enough to defend live.

Avoid combining every project thread into one talk. Data pipelines, interpretability, SOTA benchmarking, NoProp, ReD sampling, and the Streamlit UI are too much for a single conference slot.

## What Is Missing Before Submission

The project is submission-worthy now as a case study, but the proposal will be much stronger with:

- Hard benchmark numbers: termination rate, perplexity, critic accuracy, ReD coverage gains, runtime, and memory footprint.
- A clean demo path: the Streamlit playground should run locally with cached examples and no live-training dependency.
- A single story arc: one audience problem, one technical approach, one measured outcome.
- Evidence-backed claims: every major claim should point to a chart, table, run artifact, or reproducible script.
- Audience translation: biology should motivate the problem, but the lesson should be useful to data engineers and ML practitioners.

## Recommended Final Positioning

Submit it as a practical AI/data engineering talk:

> We built a compact codon-level GPT pipeline for bacterial genomic sequences, including data preparation, training, analysis, generation, and critic-based validation. The talk shows what worked, what failed, and which data and engineering choices mattered most under tight hardware constraints.

That framing is credible, useful to the hayaData audience, and easier to defend than a broad scientific breakthrough claim.
