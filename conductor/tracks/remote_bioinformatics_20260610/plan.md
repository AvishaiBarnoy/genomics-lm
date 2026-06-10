# Remote Bioinformatics Integrations Plan

## Phase 1: Client & Caching Layer
- **Task 1.1: Local Bio Cache**
  - Implement a persistent database/file cache for query hashes.
- **Task 1.2: NCBI / EBI Clients**
  - Implement rate-limited HTTP query routines in `src/eval/remote_bio.py` with error/timeout handshakes.

## Phase 2: UI Integration
- **Task 2.1: UI Annotation Widget**
  - Add an "Annotate via BLAST" button to the sequence generation interface in [web_dashboard.py](file:///Users/User/github/genomics-lm/scripts/web_dashboard.py).
- **Task 2.2: Visual Alignment Plots**
  - Render search results (hit names, species, E-values) as a clean table/visualization.

## Phase 3: Verification
- **Task 3.1: Mock Network Tests**
  - Write unit tests utilizing unittest.mock to verify API request generation and local cache hits/misses.
