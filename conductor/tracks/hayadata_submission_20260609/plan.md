# hayaData 2026 Submission Implementation Plan

This plan details the steps required to finalize the hayaData 2026 submission and prepare the presentation materials.

---

## Phase 1: CFP Preparation & Submission
- [ ] **Task 1.1:** Select the primary submission track (e.g., MLOps/Practical ML vs. Data Science).
- [ ] **Task 1.2:** Finalize abstract text, takeaways, and description on Sessionize.
- [ ] **Task 1.3:** Prepare a professional speaker bio highlighting PhD research context and frugal AI constraints.
- [ ] **Task 1.4:** Submit the proposal before the hayaData 2026 CFP closes.

---

## Phase 2: Slide Deck Development
- [ ] **Task 2.1:** Review the draft slide deck at [conference/slides.md](file:///Users/User/github/genomics-lm/conference/slides.md).
- [ ] **Task 2.2:** Update slides with technical graphics/diagrams representing:
  - The Generator-Critic loop (DNA Codons -> translation -> Protein Critic).
  - The Reset-and-Discard (ReD) sampling comparison (Standard vs. ReD round-robin).
  - Physical DNA shape correlation charts.
- [ ] **Task 2.3:** Compile slides using Marp to PDF/HTML formats for sharing.

---

## Phase 3: Live Demo Polish
- [ ] **Task 3.1:** Polish the Streamlit app [scripts/web_dashboard.py](file:///Users/User/github/genomics-lm/scripts/web_dashboard.py) to ensure it runs cleanly on local hardware without unexpected delays.
- [ ] **Task 3.2:** Pre-cache test sequences to prevent network/model latency issues during the presentation.

---

## Phase 4: Rehearsal & Dry-Run
- [ ] **Task 4.1:** Conduct a dry-run presentation focusing on:
  - Pacing (timing checks).
  - Explaining the biological intuition simply for a general data engineering/data science audience.
  - Explaining the MLOps wins (SDPA, caching, decoupled architecture).
- [ ] **Task 4.2:** Solicit peer feedback on slide clarity and technical depth.
