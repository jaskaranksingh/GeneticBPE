"""
Utility functions for GeneticBPE.
"""

import numpy as np
from typing import List, Dict, Set
import re

def validate_sequence(sequence: str) -> bool:
    """
    Validate if a sequence contains only valid RNA nucleotides.
    
    Args:
        sequence: Input sequence to validate
        
    Returns:
        bool: True if sequence is valid, False otherwise
    """
    valid_nucleotides = set('AUCG')
    return all(nuc in valid_nucleotides for nuc in sequence)

def calculate_compression_ratio(original: str, tokenized: List[str]) -> float:
    """
    Calculate the compression ratio between original and tokenized sequences.
    
    Args:
        original: Original sequence
        tokenized: List of tokens
        
    Returns:
        float: Compression ratio
    """
    return len(original) / len(tokenized)

def calculate_motif_preservation(
    sequence: str,
    tokenized: List[str],
    motifs: Dict[str, str]
) -> float:
    """
    Calculate the percentage of motifs preserved in tokenization.
    
    Args:
        sequence: Original sequence
        tokenized: List of tokens
        motifs: Dictionary of motif names to sequences
        
    Returns:
        float: Percentage of motifs preserved (0-100)
    """
    total_motifs = 0
    preserved_motifs = 0
    
    for motif in motifs.values():
        if motif in sequence:
            total_motifs += 1
            # Check if motif is preserved in tokenized sequence
            tokenized_str = ''.join(tokenized)
            if motif in tokenized_str:
                preserved_motifs += 1
    
    return (preserved_motifs / total_motifs * 100) if total_motifs > 0 else 100.0

def get_token_statistics(
    sequences: List[str],
    tokenizer
) -> Dict[str, float]:
    """
    Calculate various statistics about tokenization.
    
    Args:
        sequences: List of input sequences
        tokenizer: Trained tokenizer
        
    Returns:
        Dict containing statistics
    """
    stats = {
        'avg_tokens_per_seq': 0,
        'compression_ratio': 0,
        'vocab_usage': 0
    }
    
    total_tokens = 0
    total_original_length = 0
    used_tokens = set()
    
    for seq in sequences:
        tokens = tokenizer.encode(seq)
        total_tokens += len(tokens)
        total_original_length += len(seq)
        used_tokens.update(tokens)
    
    n_sequences = len(sequences)
    stats['avg_tokens_per_seq'] = total_tokens / n_sequences
    stats['compression_ratio'] = total_original_length / total_tokens
    stats['vocab_usage'] = len(used_tokens) / len(tokenizer.vocab) * 100
    
    return stats

def visualize_tokenization(
    sequence: str,
    tokenized: List[str],
    motifs: Dict[str, str]
) -> str:
    """
    Create a visualization of how a sequence is tokenized.
    
    Args:
        sequence: Original sequence
        tokenized: List of tokens
        motifs: Dictionary of motif names to sequences
        
    Returns:
        str: Visualization string
    """
    # Create a mapping of positions to token boundaries
    boundaries = set()
    pos = 0
    for token in tokenized:
        boundaries.add(pos)
        pos += len(token)
    boundaries.add(len(sequence))
    
    # Create visualization
    vis = []
    vis.append(sequence)
    vis.append(''.join('|' if i in boundaries else ' ' for i in range(len(sequence))))
    
    # Add motif annotations
    for name, motif in motifs.items():
        if motif in sequence:
            start = sequence.find(motif)
            end = start + len(motif)
            vis.append(' ' * start + '^' * len(motif) + ' ' * (len(sequence) - end))
            vis.append(' ' * start + name + ' ' * (len(sequence) - end - len(name)))
    
    return '\n'.join(vis) 