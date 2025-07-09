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
    def __init__(self, motif_file: Optional[str] = None, core_rules: bool = True):
        # Initialize motif DataFrame with columns and proper dtypes
        self.motifs_df = pd.DataFrame(columns=[
            'motif_id', 'sequence', 'category', 'description', 'frequency', 'species', 'length',
            'discovery_date', 'last_updated', 'confidence_score', 'references'])
        self.motifs_df = self.motifs_df.astype({
            'motif_id': 'str', 'sequence': 'str', 'category': 'str', 'description': 'str',
            'frequency': 'int64', 'species': 'str', 'length': 'int64',
            'discovery_date': 'datetime64[ns]', 'last_updated': 'datetime64[ns]',
            'confidence_score': 'float64', 'references': 'object'})
        if core_rules:
            self._initialize_core_rules()
        if motif_file:
            self.load_motifs(motif_file)
        self.motif_file = motif_file

    def _initialize_core_rules(self):
        """Initialize only core rules (e.g., seed/conserved motifs)."""
        core_motifs = [
            {'motif_id': 'seed_2_7', 'sequence': 'NNNNNN', 'category': 'seed', 'description': '6-mer seed region', 'frequency': 0, 'length': 6},
            {'motif_id': 'conserved_1', 'sequence': 'UGUGA', 'category': 'conserved', 'description': 'Common conserved sequence', 'frequency': 0, 'length': 5},
            # Add more core rules as needed
        ]
        for motif in core_motifs:
            self.add_motif(**motif)

    def add_motif(self, **kwargs):
        """Add a new motif and save to CSV if file is set."""
        new_motif = pd.DataFrame([{**kwargs,
            'discovery_date': pd.Timestamp.now(),
            'last_updated': pd.Timestamp.now(),
            'references': kwargs.get('references', [])
        }])
        new_motif = new_motif.astype(self.motifs_df.dtypes.to_dict())
        self.motifs_df = pd.concat([self.motifs_df, new_motif], ignore_index=True)
        if self.motif_file:
            self.save_motifs(self.motif_file)

    def update_motif(self, motif_id: str, **kwargs):
        """Update an existing motif and save to CSV if file is set."""
        idx = self.motifs_df[self.motifs_df['motif_id'] == motif_id].index
        for key, value in kwargs.items():
            self.motifs_df.loc[idx, key] = value
        self.motifs_df.loc[idx, 'last_updated'] = pd.Timestamp.now()
        if self.motif_file:
            self.save_motifs(self.motif_file)

    def remove_motif(self, motif_id: str):
        """Remove a motif and save to CSV if file is set."""
        self.motifs_df = self.motifs_df[self.motifs_df['motif_id'] != motif_id]
        if self.motif_file:
            self.save_motifs(self.motif_file)

    def save_motifs(self, path: str = None):
        """Save motifs to a CSV file."""
        if not path:
            path = self.motif_file
        if path:
            self.motifs_df.to_csv(path, index=False)

    def load_motifs(self, path: str):
        """Load motifs from a CSV file."""
        loaded_df = pd.read_csv(path)
        loaded_df = loaded_df.astype(self.motifs_df.dtypes.to_dict())
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