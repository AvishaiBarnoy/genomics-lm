# Remote Bioinformatics Integrations Specification

## Overview
Integrate remote bioinformatics APIs (NCBI BLAST, EBI services) into the model querying ecosystem. This provides auto-annotations for sequences generated interactively in the web dashboard, verifying their similarity to known protein families and taxonomy.

## Requirements
- **API Clients:** Implement lightweight, asynchronous clients for NCBI BLAST and EBI search APIs.
- **Local Caching:** Store all remote queries and results in a local Sqlite database or JSON cache to avoid redundant network requests.
- **Rate-Limiting & Safety:** Implement client-side rate-limits (e.g., maximum 3 requests per minute) and a fallback local-only mode.
- **Opt-In Behavior:** Keep remote features strictly disabled by default (local-first philosophy).
