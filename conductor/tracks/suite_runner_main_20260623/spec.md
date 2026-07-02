# Suite Runner Main.sh Evolution Spec

## Overview
Evolve `main.sh` from a CodonLM-specific wrapper into an explicit, dispatcher-level runner for both CodonLM and ProteinLM workflows.

## Requirements
- Support trainer-type dispatching without accidental crossover of data loading or evaluations.
- Retain legacy backward compatibility with standard arguments from previous runs.

## Success Criteria
1. **Dispatch Correctness:** `main.sh` successfully dispatches CodonLM and ProteinLM runs to their respective trainers without accidental crossover.
2. **Backward Compatibility:** Legacy arguments (e.g. Stage 1/2 runs) resolve exactly to their previous behaviors.
