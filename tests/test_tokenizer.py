"""
Tests for GeneticBPE tokenizer.
"""

import unittest
from genetic_bpe import GeneticBPETokenizer
from genetic_bpe.motif_span_manager import MotifSpanManager
from genetic_bpe.motif_bank import MotifBank
import os
import json

class TestGeneticBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.sequences = [
            "UGUGAUAUGCAUGCAUGC",
            "AUGCAUGCAUGCAUGCAU",
            "UGCUUGAUAUGCAUGCAU",
        ]
        self.config_path = "genetic_bpe/genetic_bpe_config.json"
        self.tokenizer = GeneticBPETokenizer(
            vocab_size=512,
            min_freq=2,
            config_path=self.config_path
        )
        self.motif_manager = MotifSpanManager(self.tokenizer.motif_bank)

    def test_config_loading(self):
        """Test that lambda/mu are loaded from config file."""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        self.assertEqual(self.tokenizer.motif_weight, config['motif_weight'])
        self.assertEqual(self.tokenizer.penalty_weight, config['penalty_weight'])

    def test_config_reload(self):
        """Test that config reload updates lambda/mu."""
        # Change config
        with open(self.config_path, 'w') as f:
            json.dump({"motif_weight": 3.0, "penalty_weight": 20.0}, f)
        self.tokenizer.reload_config()
        self.assertEqual(self.tokenizer.motif_weight, 3.0)
        self.assertEqual(self.tokenizer.penalty_weight, 20.0)
        # Restore config
        with open(self.config_path, 'w') as f:
            json.dump({"motif_weight": 2.5, "penalty_weight": 10.0}, f)
        self.tokenizer.reload_config()

    def test_tokenizer_training(self):
        self.tokenizer.train(self.sequences)
        self.assertGreater(len(self.tokenizer.vocab), 0)
        self.assertGreater(len(self.tokenizer.merges), 0)

    def test_motif_boundary_preservation(self):
        self.tokenizer.train(self.sequences)
        test_seq = self.sequences[0]
        motif_spans = self.motif_manager.get_motif_spans(test_seq)
        tokens = self.tokenizer.tokenize(test_seq)
        # Check that no token crosses a motif boundary
        pos = 0
        for token in tokens:
            next_pos = pos + len(token)
            for m_start, m_end, _ in motif_spans:
                self.assertFalse(pos < m_end and next_pos > m_end, f"Token {token} crosses motif boundary at {m_end}")
            pos = next_pos

if __name__ == '__main__':
    unittest.main() 