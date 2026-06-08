#!/usr/bin/env python3
"""
Stage 3: UniProt Metadata Fetcher
Maps NCBI Protein IDs (RefSeq) to UniProtKB and extracts Pfam, GO, and EC annotations.
Handles batching, polling, and pagination for large datasets.

Usage:
  python scripts/fetch_uniprot_metadata.py --input data/processed/protein_pfam_labels.json
"""

import argparse
import json
import time
import requests
import pandas as pd
from pathlib import Path

API_URL = "https://rest.uniprot.org"
# Mapping from NCBI RefSeq Protein to UniProt Accession
FROM_DB = "RefSeq_Protein"
TO_DB = "UniProtKB"

# Fields we want to extract
# See: https://www.uniprot.org/help/return_fields
FIELDS = "accession,id,xref_pfam,go,ec,length,gene_names"

def submit_id_mapping(ids):
    response = requests.post(
        f"{API_URL}/idmapping/run",
        data={"from": FROM_DB, "to": TO_DB, "ids": ",".join(ids)},
    )
    response.raise_for_status()
    return response.json()["jobId"]

def check_id_mapping_status(job_id):
    while True:
        response = requests.get(f"{API_URL}/idmapping/status/{job_id}")
        response.raise_for_status()
        data = response.json()
        if "results" in data or "jobStatus" not in data:
            return data
        
        # Job still running
        status = data.get("jobStatus", "RUNNING")
        print(f"[*] Job {job_id} is {status}. Waiting 5 seconds...")
        time.sleep(5)

def get_next_link(headers):
    if "Link" in headers:
        parts = headers["Link"].split(";")
        if 'rel="next"' in parts[1]:
            return parts[0].strip("<>")
    return None

def fetch_results(job_id):
    # For ID mapping with enrichment, UniProt provides a specific endpoint
    # that allows us to get columns for the TARGET entries (UniProtKB).
    url = f"{API_URL}/idmapping/uniprotkb/results/{job_id}"
    tsv_fields = "accession,id,xref_pfam,go,ec,length"
    params = {"fields": tsv_fields, "format": "tsv", "size": 500}
    
    all_lines = []
    print(f"[*] Fetching enriched results from {url}...")
    while url:
        response = requests.get(url, params=params if "?" not in url else None)
        response.raise_for_status()
        
        lines = response.text.strip().split("\n")
        if not lines or len(lines) < 1:
            break

        if not all_lines:
            all_lines.extend(lines) # Include header
        else:
            if len(lines) > 1:
                all_lines.extend(lines[1:]) # Skip header
            
        url = get_next_link(response.headers)
        if url:
            print(f"[*] Fetching next page: {len(all_results)} results so far...")
            
    return all_lines

def parse_uniprot_results(results_lines):
    """Parses TSV lines into a clean flat list."""
    if not results_lines: return []
    
    import io
    import csv
    # Use csv module to handle the TSV
    f = io.StringIO("\n".join(results_lines))
    reader = csv.DictReader(f, delimiter="\t")
    
    parsed = []
    # Log the headers once for debugging
    headers = reader.fieldnames
    print(f"[*] Parsing UniProt results with headers: {headers}")

    for row in reader:
        # Map the TSV columns to our clean output format
        # In ID mapping TSV: 'From' is the source NCBI ID, 'Entry' is UniProt AC
        parsed.append({
            "ncbi_id": row.get("From"),
            "uniprot_ac": row.get("Entry") or row.get("Entry"),
            "entry_name": row.get("Entry Name") or row.get("Entry name"),
            "length": row.get("Length"),
            "pfam": row.get("Cross-reference (Pfam)") or row.get("Pfam"),
            "go": row.get("Gene Ontology (GO)") or row.get("Gene ontology (GO)"),
            "ec": row.get("EC number") or row.get("EC Number")
        })
    return parsed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSON file containing protein IDs")
    ap.add_argument("--output", default="data/processed/uniprot_metadata.csv")
    ap.add_argument("--batch_size", type=int, default=1000)
    ap.add_argument("--limit", type=int, help="Limit number of IDs for testing")
    args = ap.parse_args()

    # 1. Load IDs
    with open(args.input, "r") as f:
        id_map = json.load(f)
    
    all_ids = list(id_map.keys())
    if args.limit:
        all_ids = all_ids[:args.limit]
    
    print(f"[*] Starting metadata fetch for {len(all_ids)} IDs in batches of {args.batch_size}...")

    final_parsed = []
    
    # 2. Process in Batches
    for i in range(0, len(all_ids), args.batch_size):
        batch = all_ids[i:i+args.batch_size]
        print(f"\n[Batch {i//args.batch_size + 1}] Processing {len(batch)} IDs...")
        
        try:
            job_id = submit_id_mapping(batch)
            check_id_mapping_status(job_id)
            results = fetch_results(job_id)
            parsed = parse_uniprot_results(results)
            final_parsed.extend(parsed)
            print(f"[success] Captured {len(parsed)} mappings for this batch.")
            
            # Incremental Save
            df = pd.DataFrame(final_parsed)
            df.to_csv(args.output, index=False)
            
        except Exception as e:
            print(f"[!] Error in batch {i}: {e}")
            continue

    print(f"\n[DONE] Saved total {len(final_parsed)} protein records to {args.output}")

if __name__ == "__main__":
    main()
