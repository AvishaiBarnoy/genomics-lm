
import unittest
from src.protein_lm.tokenizer import ProteinTokenizer

class TestProteinTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = ProteinTokenizer()

    def test_round_trip_sequence(self):
        sequence = "ARNDCQEGHILKMFPSTWYV"
        encoded = self.tokenizer.encode_sequence(sequence)
        decoded = self.tokenizer.decode_sequence(encoded)
        self.assertEqual(sequence, decoded)

    def test_unknown_amino_acid(self):
        sequence = "ARNDX"
        encoded = self.tokenizer.encode_sequence(sequence)
        self.assertEqual(encoded, [
            self.tokenizer.token_to_id['A'],
            self.tokenizer.token_to_id['R'],
            self.tokenizer.token_to_id['N'],
            self.tokenizer.token_to_id['D'],
            self.tokenizer.token_to_id['X']
        ])
        decoded = self.tokenizer.decode_sequence(encoded)
        self.assertEqual("ARNDX", decoded)

    def test_encode_conditions(self):
        conditions = ["<FUNC:ENZYME>", "<TOPO:TM>"]
        encoded = self.tokenizer.encode_conditions(conditions)
        self.assertEqual(encoded, [
            self.tokenizer.token_to_id["<FUNC:ENZYME>"],
            self.tokenizer.token_to_id["<TOPO:TM>"]
        ])

    def test_special_tokens(self):
        self.assertIsNotNone(self.tokenizer.bos_token_id)
        self.assertIsNotNone(self.tokenizer.eos_token_id)
        self.assertIsNotNone(self.tokenizer.pad_token_id)

    def test_vocab_size(self):
        # 20 AAs + 1 unknown + 3 special + 4 conditions
        self.assertEqual(len(self.tokenizer), 28)

if __name__ == '__main__':
    unittest.main()
