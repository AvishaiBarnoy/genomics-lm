from __future__ import annotations
from typing import Dict, List, Tuple, Optional

# Special token constants
PAD_TOKEN = "<PAD>"
BOS_CDS = "<BOS_CDS>"
EOS_CDS = "<EOS_CDS>"
UNK_TOKEN = "<UNK>"
UTR_START = "<UTR_START>"
UTR_END = "<UTR_END>"

class HybridTokenizer:
    """Tokenizer that processes DNA coding sequences (CDS) as codons

    and UTR/intergenic spacers as single nucleotides.
    Unified vocabulary of 74 tokens.
    """

    def __init__(self) -> None:
        # 1. Initialize Special Tokens
        self.special_tokens = [
            PAD_TOKEN,
            BOS_CDS,
            EOS_CDS,
            UNK_TOKEN,
            UTR_START,
            UTR_END,
        ]
        
        # 2. Initialize Codons (64 combinations of A, C, G, T)
        bases = ["A", "C", "G", "T"]
        self.codons = [b1 + b2 + b3 for b1 in bases for b2 in bases for b3 in bases]
        
        # 3. Initialize Single Nucleotides
        self.nucleotides = bases
        
        # 4. Build Vocabulary Mappings
        self.vocab = list(self.special_tokens) + list(self.codons) + list(self.nucleotides)
        self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
        self.itos = list(self.vocab)
        
        self.vocab_size = len(self.vocab)

    @staticmethod
    def reverse_complement(seq: str) -> str:
        """Returns the reverse complement of a DNA sequence."""
        comp = {
            "A": "T", "T": "A", "C": "G", "G": "C",
            "a": "t", "t": "a", "c": "g", "g": "c",
            "N": "N", "n": "n"
        }
        return "".join(comp.get(base, base) for base in reversed(seq))

    def encode(self, sequence: str, cds_intervals: List[Tuple[int, int, str]]) -> List[int]:
        """Encodes a genomic sequence into a list of hybrid token IDs.

        Args:
            sequence: Raw genomic DNA string.
            cds_intervals: List of (start, end, strand) tuples representing CDS segments.
                           Indices are 0-indexed, start is inclusive, end is exclusive.
                           Strand is '+' or '-'.

        Returns:
            List of token IDs.
        """
        seq_len = len(sequence)
        if not sequence:
            return []

        # Sort intervals to ensure left-to-right processing
        sorted_cds = sorted(cds_intervals, key=lambda x: x[0])
        
        # Validate for overlaps (not supported in base tokenizer)
        for i in range(len(sorted_cds) - 1):
            if sorted_cds[i][1] > sorted_cds[i + 1][0]:
                raise ValueError("Overlapping CDS intervals are not supported in the standard HybridTokenizer.")

        # Segment sequence into UTR and CDS chunks
        segments: List[Tuple[str, int, int, Optional[str]]] = []
        curr_idx = 0
        for start, end, strand in sorted_cds:
            if start > curr_idx:
                segments.append(("UTR", curr_idx, start, None))
            segments.append(("CDS", start, end, strand))
            curr_idx = end
        if curr_idx < seq_len:
            segments.append(("UTR", curr_idx, seq_len, None))

        token_ids: List[int] = []
        for seg_type, start, end, strand in segments:
            seg_seq = sequence[start:end].upper()
            
            if seg_type == "UTR":
                if not seg_seq:
                    continue
                # Wrap UTR in boundary tags
                token_ids.append(self.stoi[UTR_START])
                for base in seg_seq:
                    token_ids.append(self.stoi.get(base, self.stoi[UNK_TOKEN]))
                token_ids.append(self.stoi[UTR_END])
                
            elif seg_type == "CDS":
                if not seg_seq:
                    continue
                # Wrap CDS in boundary tags
                token_ids.append(self.stoi[BOS_CDS])
                
                # Orient reverse strand coding sequences
                if strand == "-":
                    coding_seq = self.reverse_complement(seg_seq)
                else:
                    coding_seq = seg_seq
                
                # Tokenize as codons
                for i in range(0, len(coding_seq) - 2, 3):
                    codon = coding_seq[i:i+3]
                    token_ids.append(self.stoi.get(codon, self.stoi[UNK_TOKEN]))
                
                token_ids.append(self.stoi[EOS_CDS])

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs into their sequence representation.

        Note: Decoded output represents the mRNA coding orientation for CDS segments.

        Args:
            token_ids: List of token IDs.

        Returns:
            Decoded DNA string.
        """
        decoded_parts: List[str] = []
        for tid in token_ids:
            if tid < 0 or tid >= self.vocab_size:
                continue
            tok = self.itos[tid]
            if tok in self.special_tokens:
                continue
            decoded_parts.append(tok)
        return "".join(decoded_parts)

    def decode_genomic(self, token_ids: List[int], cds_intervals: List[Tuple[int, int, str]]) -> str:
        """Reconstructs the original genomic DNA sequence from token IDs.

        It reverse-complements the decoded CDS segments if their interval strand is '-'.

        Args:
            token_ids: List of token IDs.
            cds_intervals: The original list of coding intervals.

        Returns:
            Reconstructed genomic DNA string matching original coordinates.
        """
        # Determine segments list
        sorted_cds = sorted(cds_intervals, key=lambda x: x[0])
        
        # We process tokens to reconstruct each segment
        token_idx = 0
        decoded_segments: List[str] = []
        
        # Build segments list structure
        # UTR segments are between CDS segments
        segments = []
        curr_idx = 0
        for start, end, strand in sorted_cds:
            if start > curr_idx:
                segments.append(("UTR", start - curr_idx, None))
            segments.append(("CDS", end - start, strand))
            curr_idx = end
        
        # Identify trailing UTR if any
        # We need to map tokens back to segments
        for seg_type, length, strand in segments:
            if seg_type == "UTR":
                # UTR is wrapped in UTR_START and UTR_END
                if token_idx < len(token_ids) and token_ids[token_idx] == self.stoi[UTR_START]:
                    token_idx += 1
                utr_bases = []
                while token_idx < len(token_ids) and token_ids[token_idx] != self.stoi[UTR_END]:
                    utr_bases.append(self.itos[token_ids[token_idx]])
                    token_idx += 1
                if token_idx < len(token_ids):
                    token_idx += 1  # consume UTR_END
                decoded_segments.append("".join(utr_bases))
                
            elif seg_type == "CDS":
                # CDS is wrapped in BOS_CDS and EOS_CDS
                if token_idx < len(token_ids) and token_ids[token_idx] == self.stoi[BOS_CDS]:
                    token_idx += 1
                cds_codons = []
                while token_idx < len(token_ids) and token_ids[token_idx] != self.stoi[EOS_CDS]:
                    cds_codons.append(self.itos[token_ids[token_idx]])
                    token_idx += 1
                if token_idx < len(token_ids):
                    token_idx += 1  # consume EOS_CDS
                
                cds_seq = "".join(cds_codons)
                if strand == "-":
                    cds_seq = self.reverse_complement(cds_seq)
                decoded_segments.append(cds_seq)
                
        # Handle remaining trailing UTR if present in tokens
        if token_idx < len(token_ids) and token_ids[token_idx] == self.stoi[UTR_START]:
            token_idx += 1
            utr_bases = []
            while token_idx < len(token_ids) and token_ids[token_idx] != self.stoi[UTR_END]:
                utr_bases.append(self.itos[token_ids[token_idx]])
                token_idx += 1
            decoded_segments.append("".join(utr_bases))

        return "".join(decoded_segments)
