import unittest
import pandas as pd
import os
from genetic_bpe.motif_bank import MotifBank
from genetic_bpe.tokenizer import GeneticBPETokenizer
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TestMotifBank(unittest.TestCase):
    def setUp(self):
        # Create test directory if it doesn't exist
        self.test_dir = "tests/test_data"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Sample miRNA sequences
        self.sample_sequences = [
            "UGUGAUACGUGAUA",  # Human miRNA-21
            "UGUGAUACGUGAUG",  # Mouse miRNA-21
            "AUGCAUGCAUGCAU",  # Conserved motif example
            "NNNNNNUGUGAUA",   # Seed region example
        ]
        
        # Initialize motif bank
        self.motif_bank = MotifBank()
        
        # Save initial motifs to CSV
        self.csv_path = os.path.join(self.test_dir, "motifs.csv")
        self.motif_bank.save_motifs(self.csv_path)
        
        # Initialize tokenizers
        self.genetic_bpe = GeneticBPETokenizer()
        
    def test_motif_discovery(self):
        """Test motif discovery and logging"""
        log_file = os.path.join(self.test_dir, f"tokenization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(log_file, 'w') as f:
            f.write("=== Tokenization Comparison Report ===\n\n")
            
            for seq in self.sample_sequences:
                f.write(f"\nProcessing sequence: {seq}\n")
                f.write("-" * 50 + "\n")
                
                # 1. GeneticBPE Tokenization
                f.write("\n1. GeneticBPE Tokenization:\n")
                tokens = self.genetic_bpe.tokenize(seq)
                f.write(f"Tokens: {tokens}\n")
                
                # Find motifs in sequence
                motifs = self.motif_bank.find_motifs_in_sequence(seq)
                f.write("\nFound motifs:\n")
                f.write(motifs.to_string())
                
                # Get statistics
                stats = self.motif_bank.get_motif_statistics()
                f.write("\n\nMotif Statistics:\n")
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
                
                f.write("\n" + "="*50 + "\n")
        
        # Verify CSV file exists and has content
        self.assertTrue(os.path.exists(self.csv_path))
        df = pd.read_csv(self.csv_path)
        self.assertGreater(len(df), 0)
        
        # Verify log file exists and has content
        self.assertTrue(os.path.exists(log_file))
        with open(log_file, 'r') as f:
            content = f.read()
            self.assertGreater(len(content), 0)

if __name__ == '__main__':
    unittest.main() 