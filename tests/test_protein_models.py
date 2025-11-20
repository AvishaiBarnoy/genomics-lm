
import unittest
import torch
from src.protein_lm.config import ProteinLMConfig, ProteinClassifierConfig
from src.protein_lm.models import ProteinConditionalTransformer, ProteinClassifier

class TestProteinModels(unittest.TestCase):

    def setUp(self):
        self.lm_config = ProteinLMConfig(
            vocab_size=30,
            n_layer=2,
            n_head=2,
            n_embd=128,
            block_size=256,
            dropout=0.1
        )
        self.classifier_config = ProteinClassifierConfig(
            vocab_size=30,
            n_layer=2,
            n_head=2,
            n_embd=128,
            block_size=256,
            dropout=0.1,
            num_classes=2
        )

    def test_lm_forward_pass(self):
        model = ProteinConditionalTransformer(self.lm_config)
        model.eval()
        
        input_ids = torch.randint(0, self.lm_config.vocab_size, (4, 100)) # batch_size=4, seq_len=100
        
        with torch.no_grad():
            logits = model(input_ids)
            
        self.assertEqual(logits.shape, (4, 100, self.lm_config.vocab_size))

    def test_classifier_forward_pass(self):
        model = ProteinClassifier(self.classifier_config)
        model.eval()
        
        input_ids = torch.randint(0, self.classifier_config.vocab_size, (4, 100))
        
        with torch.no_grad():
            logits = model(input_ids)
            
        self.assertEqual(logits.shape, (4, self.classifier_config.num_classes))

    def test_causal_mask(self):
        model = ProteinConditionalTransformer(self.lm_config)
        model.eval()
        
        input_ids = torch.randint(0, self.lm_config.vocab_size, (1, 5))
        
        # Create two versions of the input, where the second is a continuation of the first
        input_1 = input_ids[:, :-1] # (1, 4)
        input_2 = input_ids # (1, 5)
        
        # The logits for the first 4 tokens should be identical
        logits_1 = model(input_1)
        logits_2 = model(input_2)
        
        self.assertTrue(torch.allclose(logits_1, logits_2[:, :-1, :], atol=1e-6))


if __name__ == '__main__':
    unittest.main()
