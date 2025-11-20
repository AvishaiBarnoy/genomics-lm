from typing import List

class ProteinTokenizer:
    """
    A tokenizer for protein sequences and conditional tokens.
    """
    def __init__(self):
        self.amino_acids = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
        ]
        self.unknown_token = 'X'

        self.special_tokens = {
            'BOS': '<BOS>',
            'EOS': '<EOS>',
            'PAD': '<PAD>',
        }

        self.condition_tokens = {
            'FUNC_ENZYME': '<FUNC:ENZYME>',
            'FUNC_NON_ENZYME': '<FUNC:NON_ENZYME>',
            'TOPO_TM': '<TOPO:TM>',
            'TOPO_GLOBULAR': '<TOPO:GLOBULAR>',
        }

        self.vocab = (
            [self.special_tokens['PAD']] +
            [self.special_tokens['BOS']] +
            [self.special_tokens['EOS']] +
            self.amino_acids +
            [self.unknown_token] +
            list(self.condition_tokens.values())
        )

        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

    def encode_sequence(self, seq: str) -> List[int]:
        """Converts a protein sequence string into a list of token IDs."""
        return [self.token_to_id.get(aa, self.token_to_id[self.unknown_token]) for aa in seq]

    def decode_sequence(self, ids: List[int]) -> str:
        """Converts a list of token IDs back into a protein sequence string."""
        return "".join([self.id_to_token[i] for i in ids if self.id_to_token[i] not in list(self.special_tokens.values()) + list(self.condition_tokens.values())])

    def encode_conditions(self, cond_list: List[str]) -> List[int]:
        """Converts a list of condition token strings into a list of token IDs."""
        return [self.token_to_id[cond] for cond in cond_list]

    @property
    def bos_token_id(self):
        return self.token_to_id[self.special_tokens['BOS']]

    @property
    def eos_token_id(self):
        return self.token_to_id[self.special_tokens['EOS']]

    @property
    def pad_token_id(self):
        return self.token_to_id[self.special_tokens['PAD']]

    def __len__(self):
        return len(self.vocab)