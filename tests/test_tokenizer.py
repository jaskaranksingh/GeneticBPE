"""
Tests for GeneticBPE tokenizer.
"""

import unittest
from genetic_bpe import GeneticBPE
from genetic_bpe.utils import validate_sequence

class TestGeneticBPE(unittest.TestCase):
    def setUp(self):
        self.sequences = [
            "UGUGAUAUGCAUGCAUGC",
            "AUGCAUGCAUGCAUGCAU",
            "UGCUUGAUAUGCAUGCAU",
        ]
        self.tokenizer = GeneticBPE(
            vocab_size=512,
            motif_weight=2.5,
            penalty_weight=10.0
        )
    
    def test_sequence_validation(self):
        """Test sequence validation."""
        valid_seq = "AUGCAUGC"
        invalid_seq = "AUGCAUGCX"  # X is not a valid nucleotide
        
        self.assertTrue(validate_sequence(valid_seq))
        self.assertFalse(validate_sequence(invalid_seq))
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization."""
        self.assertEqual(self.tokenizer.vocab_size, 512)
        self.assertEqual(self.tokenizer.motif_weight, 2.5)
        self.assertEqual(self.tokenizer.penalty_weight, 10.0)
        self.assertEqual(len(self.tokenizer.vocab), 0)
    
    def test_tokenizer_training(self):
        """Test tokenizer training."""
        self.tokenizer.train(self.sequences)
        self.assertGreater(len(self.tokenizer.vocab), 0)
        self.assertGreater(len(self.tokenizer.merges), 0)
    
    def test_encoding_decoding(self):
        """Test sequence encoding and decoding."""
        self.tokenizer.train(self.sequences)
        test_seq = self.sequences[0]
        
        # Encode
        token_ids = self.tokenizer.encode(test_seq)
        self.assertIsInstance(token_ids, list)
        self.assertTrue(all(isinstance(id, int) for id in token_ids))
        
        # Decode
        decoded_seq = self.tokenizer.decode(token_ids)
        self.assertEqual(decoded_seq, test_seq)
    
    def test_motif_preservation(self):
        """Test motif preservation during tokenization."""
        self.tokenizer.train(self.sequences)
        test_seq = self.sequences[0]
        
        # Get motifs in sequence
        motifs = self.tokenizer.motif_bank.find_motifs_in_sequence(test_seq)
        self.assertGreater(len(motifs), 0)
        
        # Check if motifs are preserved in tokenized sequence
        token_ids = self.tokenizer.encode(test_seq)
        tokens = [self.tokenizer.id_to_token[id] for id in token_ids]
        tokenized_seq = ''.join(tokens)
        
        for motif in motifs.values():
            self.assertIn(motif, tokenized_seq)

if __name__ == '__main__':
    unittest.main() 