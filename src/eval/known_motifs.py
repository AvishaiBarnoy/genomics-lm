import json
from pathlib import Path

# A curated library of biological motifs relevant to bacterial genomics.
# Consensus sequences are provided in DNA (ACGT) format.
# These will be mapped to the model's codon-level vocabulary during analysis.

KNOWN_MOTIFS = {
    "Shine-Dalgarno": {
        "sequence": "AGGAGG",
        "type": "ribosome_binding_site",
        "description": "Recruits the ribosome to the mRNA for translation initiation.",
        "location": "5-10 bp upstream of START"
    },
    "Pribnow_Box": {
        "sequence": "TATAAT",
        "type": "promoter_10",
        "description": "Core promoter element (-10 box) that facilitates DNA melting.",
        "location": "~10 bp upstream of Transcription Start"
    },
    "TTGACA_Box": {
        "sequence": "TTGACA",
        "type": "promoter_35",
        "description": "Promoter element (-35 box) for initial RNA polymerase binding.",
        "location": "~35 bp upstream of Transcription Start"
    },
    "START_Codon": {
        "sequence": "ATG",
        "type": "initiation",
        "description": "The most common translation start signal.",
        "location": "Start of CDS"
    },
    "Poly-U_Tract": {
        "sequence": "TTTTTT",
        "type": "terminator",
        "description": "U-rich region involved in Rho-independent transcription termination.",
        "location": "Downstream of STOP"
    }
}

def get_motif_library_path():
    return Path("src/eval/known_motifs.json")

def save_library():
    path = get_motif_library_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(KNOWN_MOTIFS, f, indent=4)
    print(f"[success] Saved motif library to {path}")

if __name__ == "__main__":
    save_library()
