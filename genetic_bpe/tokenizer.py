"""
GeneticBPE Tokenizer: A motif-preserving tokenizer for miRNA sequences
"""

import re
from typing import List, Dict, Set, Optional
from collections import defaultdict
from .motif_bank import MotifBank
import os
import json

class GeneticBPETokenizer:
    def __init__(self, vocab_size: int = 1000, min_freq: int = 2, config_path: str = None):
        """
        Initialize the GeneticBPE tokenizer.
        Args:
            vocab_size: Maximum size of the vocabulary
            min_freq: Minimum frequency for a token to be included in vocabulary
            config_path: Path to config file with lambda and mu
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), 'genetic_bpe_config.json')
        self.motif_weight = 2.5
        self.penalty_weight = 10.0
        self._load_config()
        self.motif_bank = MotifBank()
        self.vocab = set()
        self.merges = {}
        self.token_frequencies = defaultdict(int)

    def _load_config(self):
        """Load lambda and mu from config file if present."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            self.motif_weight = config.get('motif_weight', self.motif_weight)
            self.penalty_weight = config.get('penalty_weight', self.penalty_weight)

    def reload_config(self):
        """Reload config at runtime."""
        self._load_config()

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

    def _get_motif_spans(self, sequence: str):
        return self.motif_bank.get_motif_spans(sequence)

    def _get_pairs_with_scores(self, sequences: list) -> dict:
        """Return a dict of pairs and their motif-aware merge scores."""
        pair_stats = defaultdict(lambda: {'freq': 0, 'bonus': 0, 'penalty': 0})
        for seq in sequences:
            tokens = list(seq)
            motif_spans = self._get_motif_spans(seq)
            for i in range(len(tokens) - 1):
                pair = tokens[i] + tokens[i+1]
                pair_start = i
                pair_end = i + 2
                pair_stats[pair]['freq'] += 1
                if self.motif_bank.is_pair_inside_motif(pair_start, pair_end, motif_spans):
                    pair_stats[pair]['bonus'] += 1
                if self.motif_bank.is_pair_crossing_motif_boundary(pair_start, pair_end, motif_spans):
                    pair_stats[pair]['penalty'] += 1
        # Compute scores
        scores = {}
        for pair, stats in pair_stats.items():
            score = stats['freq'] + self.motif_weight * stats['bonus'] - self.penalty_weight * stats['penalty']
            scores[pair] = score
        return scores

    def train(self, sequences: list):
        # Count token frequencies
        for seq in sequences:
            tokens = list(seq)
            for token in tokens:
                self.token_frequencies[token] += 1
        self.vocab = {token for token, freq in self.token_frequencies.items() if freq >= self.min_freq}
        # Learn merges
        merges = {}
        current_vocab = set(self.vocab)
        corpus = [list(seq) for seq in sequences]
        while len(current_vocab) < self.vocab_size:
            pair_scores = self._get_pairs_with_scores([''.join(tokens) for tokens in corpus])
            if not pair_scores:
                break
            best_pair = max(pair_scores.items(), key=lambda x: x[1])[0]
            # If best_pair has negative score, stop
            if pair_scores[best_pair] <= 0:
                break
            # Merge best_pair in all sequences, but skip merges that would split motifs
            new_corpus = []
            for seq in corpus:
                i = 0
                new_seq = []
                motif_spans = self._get_motif_spans(''.join(seq))
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] + seq[i+1] == best_pair:
                        # Check if merge is allowed (does not split motif)
                        if self.motif_bank.is_pair_crossing_motif_boundary(i, i+2, motif_spans):
                            new_seq.append(seq[i])
                            i += 1
                        else:
                            new_seq.append(best_pair)
                            i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_corpus.append(new_seq)
            corpus = new_corpus
            merges[best_pair] = len(merges)
            current_vocab.add(best_pair)
        self.merges = merges
        self.vocab = current_vocab
    
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