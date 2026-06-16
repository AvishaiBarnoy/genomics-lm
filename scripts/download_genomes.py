#!/usr/bin/env python3
"""
Downloads diverse bacterial reference genomes from the NCBI FTP server.
Creates a taxonomically diverse dataset covering multiple phyla/families.
"""

import os
import urllib.request
import re
import gzip
import shutil
from pathlib import Path

# Accessions representing diverse bacterial families and phyla
GENOMES = {
    "Vibrio_cholerae": "GCF_000006745.1",
    "Lactiplantibacillus_plantarum": "GCF_000203855.3",
    "Streptococcus_pneumoniae": "GCF_000007045.1",
    "Helicobacter_pylori": "GCF_000008525.1",
    "Caulobacter_crescentus": "GCF_000006905.1",
    "Synechocystis_sp": "GCF_000009725.1",
    "Mycoplasma_pneumoniae": "GCF_000027325.1",
    "Chlamydia_trachomatis": "GCF_000008725.1",
    "Borrelia_burgdorferi": "GCF_000008685.2",
    "Bacteroides_thetaiotaomicron": "GCF_000011065.1",
    "Clostridioides_difficile": "GCF_000011965.2",
    "Neisseria_meningitidis": "GCF_000008805.1",
    "Campylobacter_jejuni": "GCF_000006925.2",
    "Thermus_thermophilus": "GCF_000009125.1",
    "Geobacter_sulfurreducens": "GCF_000007985.2"
}

def resolve_ncbi_ftp_path(accession: str) -> str:
    # Accession format: GCF_000006745.1
    match = re.match(r"GCF_(\d+)\.\d+", accession)
    if not match:
        raise ValueError(f"Invalid RefSeq accession format: {accession}")

    digits = match.group(1)
    # Pad to 9 digits, e.g. 6745 -> 000006745
    padded = digits.zfill(9)
    part1 = padded[0:3]
    part2 = padded[3:6]
    part3 = padded[6:9]

    # Path to GCF assembly folder on NCBI FTP
    base_url = f"https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/{part1}/{part2}/{part3}/"

    # Fetch folder index to find the full folder name (which contains the assembly name suffix)
    print(f"[*] Resolving folder listing at: {base_url}")
    req = urllib.request.Request(
        base_url,
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req) as response:
        html = response.read().decode("utf-8")

    # Search for links containing the accession
    pattern = rf'href="({accession}_[^/]+)/"'
    folders = re.findall(pattern, html)
    if not folders:
        raise RuntimeError(f"Could not locate assembly folder for {accession} in directory listing.")

    folder_name = folders[0]
    download_url = f"{base_url}{folder_name}/{folder_name}_genomic.gbff.gz"
    return download_url

def download_genome(name: str, accession: str, dest_dir: Path) -> Path:
    dest_path = dest_dir / f"{accession}.gbff"
    if dest_path.exists():
        print(f"[+] {name} ({accession}) already exists. Skipping download.")
        return dest_path

    try:
        url = resolve_ncbi_ftp_path(accession)
        print(f"[*] Downloading {name} from: {url}")

        temp_gz = dest_dir / f"{accession}.gbff.gz"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(temp_gz, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

        # Decompress gzip
        print(f"[*] Decompressing {temp_gz}...")
        with gzip.open(temp_gz, 'rb') as f_in, open(dest_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        temp_gz.unlink() # remove gz file
        print(f"[+] Successfully saved {dest_path}")
        return dest_path
    except Exception as err:
        print(f"[!] Failed to download {name} ({accession}): {err}")
        return None

def main():
    dest_dir = Path("data/raw/expanded")
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"===========================================================")
    print(f"Starting Download of {len(GENOMES)} Diverse Reference Genomes")
    print(f"Target Directory: {dest_dir}")
    print(f"===========================================================")

    success_count = 0
    for name, accession in GENOMES.items():
        res = download_genome(name, accession, dest_dir)
        if res:
            success_count += 1

    print(f"\n===========================================================")
    print(f"Download completed: {success_count}/{len(GENOMES)} genomes downloaded.")
    print(f"===========================================================")

if __name__ == "__main__":
    main()
