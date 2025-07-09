import requests
import json
import os
from datetime import datetime

class TokenizerAPITester:
    def __init__(self):
        self.base_url = "http://localhost:5000"  
        self.test_dir = "tests/test_data"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Sample sequences for testing
        self.sample_sequences = [
            "UGUGAUACGUGAUA",  # Human miRNA-21
            "UGUGAUACGUGAUG",  # Mouse miRNA-21
            "AUGCAUGCAUGCAU",  # Conserved motif example
            "NNNNNNUGUGAUA",   # Seed region example
        ]
    
    def test_tokenization(self):
        """Test tokenization endpoints"""
        log_file = os.path.join(self.test_dir, f"api_test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(log_file, 'w') as f:
            f.write("=== API Tokenization Test Report ===\n\n")
            
            for seq in self.sample_sequences:
                f.write(f"\nProcessing sequence: {seq}\n")
                f.write("-" * 50 + "\n")
                
                # Test GeneticBPE tokenization
                try:
                    response = requests.post(
                        f"{self.base_url}/tokenize",
                        json={"sequence": seq, "tokenizer": "genetic_bpe"}
                    )
                    f.write("\n1. GeneticBPE API Response:\n")
                    f.write(f"Status Code: {response.status_code}\n")
                    f.write(f"Response: {json.dumps(response.json(), indent=2)}\n")
                except Exception as e:
                    f.write(f"Error: {str(e)}\n")
                
                # Test motif discovery
                try:
                    response = requests.post(
                        f"{self.base_url}/discover_motifs",
                        json={"sequence": seq}
                    )
                    f.write("\n2. Motif Discovery API Response:\n")
                    f.write(f"Status Code: {response.status_code}\n")
                    f.write(f"Response: {json.dumps(response.json(), indent=2)}\n")
                except Exception as e:
                    f.write(f"Error: {str(e)}\n")
                
                f.write("\n" + "="*50 + "\n")
        
        return log_file

if __name__ == "__main__":
    tester = TokenizerAPITester()
    log_file = tester.test_tokenization()
    print(f"Test log saved to: {log_file}") 