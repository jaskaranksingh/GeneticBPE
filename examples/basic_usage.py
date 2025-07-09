"""
Basic example of using GeneticBPE for miRNA sequence tokenization.
"""

from genetic_bpe import GeneticBPE
from genetic_bpe.utils import (
    validate_sequence,
    calculate_compression_ratio,
    calculate_motif_preservation,
    visualize_tokenization
)

def main():
    # Example miRNA sequences
    sequences = [
        "UGUGAUAUGCAUGCAUGC",
        "AUGCAUGCAUGCAUGCAU",
        "UGCUUGAUAUGCAUGCAU",
    ]
    
    # Validate sequences
    for seq in sequences:
        if not validate_sequence(seq):
            print(f"Invalid sequence: {seq}")
            return
    
    # Initialize and train tokenizer
    tokenizer = GeneticBPE(
        vocab_size=512,
        motif_weight=2.5,
        penalty_weight=10.0
    )
    
    print("Training tokenizer...")
    tokenizer.train(sequences)
    
    # Test tokenization
    test_sequence = sequences[0]
    print(f"\nTesting tokenization on sequence: {test_sequence}")
    
    # Encode sequence
    token_ids = tokenizer.encode(test_sequence)
    tokens = [tokenizer.id_to_token[id] for id in token_ids]
    
    print(f"Tokenized sequence: {tokens}")
    
    # Calculate statistics
    compression = calculate_compression_ratio(test_sequence, tokens)
    motif_preservation = calculate_motif_preservation(
        test_sequence,
        tokens,
        tokenizer.motif_bank.get_all_motifs()
    )
    
    print(f"\nStatistics:")
    print(f"Compression ratio: {compression:.2f}x")
    print(f"Motif preservation: {motif_preservation:.1f}%")
    
    # Visualize tokenization
    print("\nTokenization visualization:")
    vis = visualize_tokenization(
        test_sequence,
        tokens,
        tokenizer.motif_bank.get_all_motifs()
    )
    print(vis)
    
    # Save tokenizer
    tokenizer.save("trained_tokenizer.json")
    print("\nTokenizer saved to 'trained_tokenizer.json'")

if __name__ == "__main__":
    main() 