"""
Enhanced motif bank for miRNA sequences using pandas for efficient data management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import json
import re

class MotifBank:
    def __init__(self, motif_file: Optional[str] = None):
        # Initialize motif DataFrame with columns and proper dtypes
        self.motifs_df = pd.DataFrame(columns=[
            'motif_id',          # Unique identifier
            'sequence',          # Motif sequence
            'category',          # Category (seed, conserved, species-specific, etc.)
            'description',       # Description of the motif
            'frequency',         # Frequency of occurrence
            'species',           # Species specificity (if applicable)
            'length',           # Length of the motif
            'discovery_date',    # When the motif was added
            'last_updated',      # Last update timestamp
            'confidence_score',  # Confidence in motif validity
            'references'         # Literature references
        ])
        
        # Set proper dtypes
        self.motifs_df = self.motifs_df.astype({
            'motif_id': 'str',
            'sequence': 'str',
            'category': 'str',
            'description': 'str',
            'frequency': 'int64',
            'species': 'str',
            'length': 'int64',
            'discovery_date': 'datetime64[ns]',
            'last_updated': 'datetime64[ns]',
            'confidence_score': 'float64',
            'references': 'object'
        })
        
        # Initialize with default motifs
        self._initialize_default_motifs()
        
        # Statistics tracking
        self.motif_stats = defaultdict(int)
        self.motif_cooccurrence = pd.DataFrame()
        
        # Load additional motifs if file provided
        if motif_file:
            self.load_motifs(motif_file)
    
    def _initialize_default_motifs(self):
        """Initialize the motif bank with default motifs."""
        default_motifs = [
            # Seed motifs (6-8mers)
            {'motif_id': 'seed_2_7', 'sequence': 'NNNNNN', 'category': 'seed',
             'description': '6-mer seed region', 'frequency': 0, 'length': 6},
            {'motif_id': 'seed_2_8', 'sequence': 'NNNNNNN', 'category': 'seed',
             'description': '7-mer seed region', 'frequency': 0, 'length': 7},
            {'motif_id': 'seed_2_9', 'sequence': 'NNNNNNNN', 'category': 'seed',
             'description': '8-mer seed region', 'frequency': 0, 'length': 8},
            
            # Conserved motifs
            {'motif_id': 'conserved_1', 'sequence': 'UGUGA', 'category': 'conserved',
             'description': 'Common conserved sequence', 'frequency': 0, 'length': 5},
            {'motif_id': 'conserved_2', 'sequence': 'AUGCA', 'category': 'conserved',
             'description': 'Another conserved sequence', 'frequency': 0, 'length': 5},
            {'motif_id': 'conserved_3', 'sequence': 'UGUGAUA', 'category': 'conserved',
             'description': 'Extended conserved sequence', 'frequency': 0, 'length': 7},
            {'motif_id': 'conserved_4', 'sequence': 'AGCUAC', 'category': 'conserved',
             'description': 'Conserved seed sequence', 'frequency': 0, 'length': 6},
            
            # Human-specific motifs
            {'motif_id': 'hsa_1', 'sequence': 'UGUGAUA', 'category': 'species-specific',
             'description': 'Human-specific motif', 'frequency': 0, 'species': 'human', 'length': 7},
            {'motif_id': 'hsa_2', 'sequence': 'AGCUACAU', 'category': 'species-specific',
             'description': 'Human-specific seed motif', 'frequency': 0, 'species': 'human', 'length': 8},
            {'motif_id': 'hsa_3', 'sequence': 'UAGCAGC', 'category': 'species-specific',
             'description': 'Human-specific conserved motif', 'frequency': 0, 'species': 'human', 'length': 7},
            
            # Mouse-specific motifs
            {'motif_id': 'mmu_1', 'sequence': 'UGUGAUG', 'category': 'species-specific',
             'description': 'Mouse-specific motif', 'frequency': 0, 'species': 'mouse', 'length': 7},
            {'motif_id': 'mmu_2', 'sequence': 'AGCUACAG', 'category': 'species-specific',
             'description': 'Mouse-specific seed motif', 'frequency': 0, 'species': 'mouse', 'length': 8},
            {'motif_id': 'mmu_3', 'sequence': 'UAGCAGU', 'category': 'species-specific',
             'description': 'Mouse-specific conserved motif', 'frequency': 0, 'species': 'mouse', 'length': 7},
            
            # Structural motifs
            {'motif_id': 'struct_1', 'sequence': 'UGUGAUACGUGAUA', 'category': 'structural',
             'description': 'Complete miRNA-21 structure', 'frequency': 0, 'length': 14},
            {'motif_id': 'struct_2', 'sequence': 'UAGCAGCACGUAAAUAUUGGCG', 'category': 'structural',
             'description': 'Complete miRNA-16 structure', 'frequency': 0, 'length': 22},
            {'motif_id': 'struct_3', 'sequence': 'UUAAUGCUAAUCGUGAUAGGGGU', 'category': 'structural',
             'description': 'Complete miRNA-155 structure', 'frequency': 0, 'length': 23},
            
            # Common patterns
            {'motif_id': 'pattern_1', 'sequence': 'UGUGA', 'category': 'pattern',
             'description': 'Common 5-mer pattern', 'frequency': 0, 'length': 5},
            {'motif_id': 'pattern_2', 'sequence': 'AGCUAC', 'category': 'pattern',
             'description': 'Common 6-mer pattern', 'frequency': 0, 'length': 6},
            {'motif_id': 'pattern_3', 'sequence': 'UAGCAGC', 'category': 'pattern',
             'description': 'Common 7-mer pattern', 'frequency': 0, 'length': 7},
            
            # Known miRNA sequences
            {'motif_id': 'mir_21', 'sequence': 'UGUGAUACGUGAUA', 'category': 'known_mirna',
             'description': 'miRNA-21 sequence', 'frequency': 0, 'length': 14},
            {'motif_id': 'mir_16', 'sequence': 'UAGCAGCACGUAAAUAUUGGCG', 'category': 'known_mirna',
             'description': 'miRNA-16 sequence', 'frequency': 0, 'length': 22},
            {'motif_id': 'mir_155', 'sequence': 'UUAAUGCUAAUCGUGAUAGGGGU', 'category': 'known_mirna',
             'description': 'miRNA-155 sequence', 'frequency': 0, 'length': 23},
            {'motif_id': 'mir_221', 'sequence': 'AGCUACAUUGUCUGCUGGGUUUC', 'category': 'known_mirna',
             'description': 'miRNA-221 sequence', 'frequency': 0, 'length': 23},
            {'motif_id': 'mir_222', 'sequence': 'AGCUACAUCUGGCUACUGGGU', 'category': 'known_mirna',
             'description': 'miRNA-222 sequence', 'frequency': 0, 'length': 21},
        ]
        
        # Add default motifs to DataFrame
        for motif in default_motifs:
            self.add_motif(**motif)
    
    def add_motif(self, motif_id: str, sequence: str, category: str, 
                  description: str = "", frequency: int = 0, species: str = None,
                  length: int = None, confidence_score: float = 1.0,
                  references: List[str] = None):
        """Add a new motif to the bank."""
        if length is None:
            length = len(sequence)
            
        new_motif = pd.DataFrame([{
            'motif_id': motif_id,
            'sequence': sequence,
            'category': category,
            'description': description,
            'frequency': int(frequency),  # Ensure frequency is int
            'species': species,
            'length': int(length),  # Ensure length is int
            'discovery_date': pd.Timestamp.now(),
            'last_updated': pd.Timestamp.now(),
            'confidence_score': float(confidence_score),  # Ensure confidence_score is float
            'references': references if references else []
        }])
        
        # Ensure proper dtypes
        new_motif = new_motif.astype({
            'motif_id': 'str',
            'sequence': 'str',
            'category': 'str',
            'description': 'str',
            'frequency': 'int64',
            'species': 'str',
            'length': 'int64',
            'discovery_date': 'datetime64[ns]',
            'last_updated': 'datetime64[ns]',
            'confidence_score': 'float64',
            'references': 'object'
        })
        
        self.motifs_df = pd.concat([self.motifs_df, new_motif], ignore_index=True)
    
    def find_motifs_in_sequence(self, sequence: str) -> pd.DataFrame:
        """Find all motifs present in a given sequence."""
        found_motifs = []
        for _, motif in self.motifs_df.iterrows():
            if motif['sequence'] in sequence:
                found_motifs.append(motif)
                # Update frequency
                self.motifs_df.loc[self.motifs_df['motif_id'] == motif['motif_id'], 'frequency'] += 1
                self.motifs_df.loc[self.motifs_df['motif_id'] == motif['motif_id'], 'last_updated'] = pd.Timestamp.now()
        
        return pd.DataFrame(found_motifs)
    
    def get_motif_statistics(self) -> Dict:
        """Get comprehensive statistics about motif usage."""
        stats = {
            'total_motifs': len(self.motifs_df),
            'motifs_by_category': self.motifs_df['category'].value_counts().to_dict(),
            'motifs_by_species': self.motifs_df['species'].value_counts().to_dict(),
            'length_distribution': self.motifs_df['length'].value_counts().to_dict(),
            'top_frequent_motifs': self.motifs_df.nlargest(10, 'frequency')[['motif_id', 'frequency']].to_dict('records'),
            'average_confidence': float(self.motifs_df['confidence_score'].mean())
        }
        return stats
    
    def discover_new_motifs(self, sequences: List[str], min_length: int = 4, 
                           max_length: int = 8, min_freq: int = 3,
                           confidence_threshold: float = 0.7):
        """Discover new motifs from a set of sequences using pandas operations."""
        # Create a DataFrame of all possible subsequences
        subsequences = []
        for seq in sequences:
            for length in range(min_length, max_length + 1):
                for i in range(len(seq) - length + 1):
                    subsequences.append({
                        'sequence': seq[i:i+length],
                        'length': length,
                        'position': i
                    })
        
        subseq_df = pd.DataFrame(subsequences)
        
        # Count frequencies
        freq_df = subseq_df['sequence'].value_counts().reset_index()
        freq_df.columns = ['sequence', 'frequency']
        
        # Filter by frequency and add new motifs
        new_motifs = freq_df[freq_df['frequency'] >= min_freq]
        
        for _, motif in new_motifs.iterrows():
            if motif['sequence'] not in self.motifs_df['sequence'].values:
                motif_id = f"discovered_{len(self.motifs_df)}"
                self.add_motif(
                    motif_id=motif_id,
                    sequence=motif['sequence'],
                    category='discovered',
                    description='Automatically discovered motif',
                    frequency=motif['frequency'],
                    length=len(motif['sequence']),
                    confidence_score=confidence_threshold
                )
    
    def save_motifs(self, path: str):
        """Save motifs to a CSV file."""
        self.motifs_df.to_csv(path, index=False)
    
    def load_motifs(self, path: str):
        """Load motifs from a CSV file."""
        loaded_df = pd.read_csv(path)
        # Ensure proper dtypes after loading
        loaded_df = loaded_df.astype({
            'motif_id': 'str',
            'sequence': 'str',
            'category': 'str',
            'description': 'str',
            'frequency': 'int64',
            'species': 'str',
            'length': 'int64',
            'discovery_date': 'datetime64[ns]',
            'last_updated': 'datetime64[ns]',
            'confidence_score': 'float64',
            'references': 'object'
        })
        self.motifs_df = pd.concat([self.motifs_df, loaded_df], ignore_index=True)
        self.motifs_df = self.motifs_df.drop_duplicates(subset=['motif_id'])
    
    def get_motifs_by_category(self, category: str) -> pd.DataFrame:
        """Get all motifs from a specific category."""
        return self.motifs_df[self.motifs_df['category'] == category]
    
    def get_motifs_by_species(self, species: str) -> pd.DataFrame:
        """Get all motifs specific to a species."""
        return self.motifs_df[self.motifs_df['species'] == species]
    
    def update_motif_confidence(self, motif_id: str, new_confidence: float):
        """Update the confidence score of a motif."""
        self.motifs_df.loc[self.motifs_df['motif_id'] == motif_id, 'confidence_score'] = float(new_confidence)
        self.motifs_df.loc[self.motifs_df['motif_id'] == motif_id, 'last_updated'] = pd.Timestamp.now()
    
    def merge_motif_banks(self, other_bank: 'MotifBank'):
        """Merge another motif bank into this one."""
        self.motifs_df = pd.concat([self.motifs_df, other_bank.motifs_df], ignore_index=True)
        self.motifs_df = self.motifs_df.drop_duplicates(subset=['motif_id'])
    
    def export_to_json(self, path: str):
        """Export motif bank to JSON format."""
        data = {
            'motifs': self.motifs_df.to_dict(orient='records'),
            'statistics': self.get_motif_statistics()
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def import_from_json(self, path: str):
        """Import motif bank from JSON format."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        new_df = pd.DataFrame(data['motifs'])
        # Ensure proper dtypes after loading
        new_df = new_df.astype({
            'motif_id': 'str',
            'sequence': 'str',
            'category': 'str',
            'description': 'str',
            'frequency': 'int64',
            'species': 'str',
            'length': 'int64',
            'discovery_date': 'datetime64[ns]',
            'last_updated': 'datetime64[ns]',
            'confidence_score': 'float64',
            'references': 'object'
        })
        self.motifs_df = pd.concat([self.motifs_df, new_df], ignore_index=True)
        self.motifs_df = self.motifs_df.drop_duplicates(subset=['motif_id']) 