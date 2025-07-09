"""
GeneticBPE Tokenizer: A motif-preserving tokenizer for miRNA sequences
"""

import re
from typing import List, Dict, Set, Optional
from collections import defaultdict
from .motif_bank import MotifBank

class GeneticBPETokenizer:
    def __init__(self, vocab_size: int = 1000, min_freq: int = 2):
        """
        Initialize the GeneticBPE tokenizer.
        
        Args:
            vocab_size: Maximum size of the vocabulary
            min_freq: Minimum frequency for a token to be included in vocabulary
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.motif_bank = MotifBank()
        self.vocab = set()
        self.merges = {}
        self.token_frequencies = defaultdict(int)
        
    def tokenize(self, sequence: str) -> List[str]:
        """
        Tokenize a miRNA sequence while preserving motifs.
        
        Args:
            sequence: Input miRNA sequence
            
        Returns:
            List of tokens
        """
        # First, find all motifs in the sequence
        motifs = self.motif_bank.find_motifs_in_sequence(sequence)
        
        # Create a list to store tokens
        tokens = []
        current_pos = 0
        
        # Sort motifs by position and length
        motif_positions = []
        for _, motif in motifs.iterrows():
            start_pos = sequence.find(motif['sequence'])
            if start_pos != -1:
                motif_positions.append((start_pos, motif['sequence']))
        
        motif_positions.sort(key=lambda x: x[0])
        
        # Process the sequence
        for pos, motif in motif_positions:
            # Add any characters before the motif
            if pos > current_pos:
                tokens.extend(self._tokenize_subsequence(sequence[current_pos:pos]))
            
            # Add the motif as a single token
            tokens.append(motif)
            current_pos = pos + len(motif)
        
        # Add any remaining characters
        if current_pos < len(sequence):
            tokens.extend(self._tokenize_subsequence(sequence[current_pos:]))
        
        return tokens
    
    def _tokenize_subsequence(self, subsequence: str) -> List[str]:
        """
        Tokenize a subsequence using BPE.
        
        Args:
            subsequence: Input subsequence
            
        Returns:
            List of tokens
        """
        # Initialize with individual characters
        tokens = list(subsequence)
        
        # Apply BPE merges
        while len(tokens) > 1:
            # Find the most frequent pair
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
                
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            if best_pair not in self.merges:
                break
                
            # Merge the pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] + tokens[i+1] == best_pair:
                    new_tokens.append(best_pair)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens
    
    def _get_pairs(self, tokens: List[str]) -> Dict[str, int]:
        """
        Get frequency of adjacent pairs in tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dictionary of pair frequencies
        """
        pairs = defaultdict(int)
        for i in range(len(tokens) - 1):
            pair = tokens[i] + tokens[i+1]
            pairs[pair] += 1
        return pairs
    
    def train(self, sequences: List[str]):
        """
        Train the tokenizer on a list of sequences.
        
        Args:
            sequences: List of miRNA sequences
        """
        # Count token frequencies
        for seq in sequences:
            tokens = self.tokenize(seq)
            for token in tokens:
                self.token_frequencies[token] += 1
        
        # Build vocabulary
        self.vocab = {token for token, freq in self.token_frequencies.items() 
                     if freq >= self.min_freq}
        
        # Learn merges
        while len(self.vocab) < self.vocab_size:
            # Find most frequent pair
            pairs = defaultdict(int)
            for seq in sequences:
                tokens = self.tokenize(seq)
                for i in range(len(tokens) - 1):
                    pair = tokens[i] + tokens[i+1]
                    pairs[pair] += 1
            
            if not pairs:
                break
                
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            self.merges[best_pair] = len(self.merges)
            
            # Update vocabulary
            self.vocab.add(best_pair)
    
    def save(self, path: str):
        """
        Save the tokenizer state.
        
        Args:
            path: Path to save the tokenizer
        """
        state = {
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'vocab': list(self.vocab),
            'merges': self.merges,
            'token_frequencies': dict(self.token_frequencies)
        }
        import json
        with open(path, 'w') as f:
            json.dump(state, f)
    
    def load(self, path: str):
        """
        Load the tokenizer state.
        
        Args:
            path: Path to load the tokenizer from
        """
        import json
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.vocab_size = state['vocab_size']
        self.min_freq = state['min_freq']
        self.vocab = set(state['vocab'])
        self.merges = state['merges']
        self.token_frequencies = defaultdict(int, state['token_frequencies']) 