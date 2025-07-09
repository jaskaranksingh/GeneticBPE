"""
MotifSpanManager: Handles motif span finding and boundary logic for GeneticBPE.
"""

from typing import List, Tuple
from .motif_bank import MotifBank

class MotifSpanManager:
    def __init__(self, motif_bank: MotifBank):
        self.motif_bank = motif_bank

    def get_motif_spans(self, sequence: str) -> List[Tuple[int, int, str]]:
        """Return a list of (start, end, motif_id) for all motifs in the sequence."""
        spans = []
        for _, motif in self.motif_bank.motifs_df.iterrows():
            motif_seq = motif['sequence']
            start = 0
            while True:
                idx = sequence.find(motif_seq, start)
                if idx == -1:
                    break
                spans.append((idx, idx + len(motif_seq), motif['motif_id']))
                start = idx + 1
        return spans

    def is_pair_inside_motif(self, pair_start: int, pair_end: int, motif_spans: List[Tuple[int, int, str]]) -> bool:
        """Check if a pair (start, end) is fully inside any motif span."""
        for m_start, m_end, _ in motif_spans:
            if pair_start >= m_start and pair_end <= m_end:
                return True
        return False

    def is_pair_crossing_motif_boundary(self, pair_start: int, pair_end: int, motif_spans: List[Tuple[int, int, str]]) -> bool:
        """Check if a pair (start, end) crosses any motif boundary."""
        for m_start, m_end, _ in motif_spans:
            if (pair_start < m_end and pair_end > m_end) or (pair_start < m_start and pair_end > m_start):
                return True
        return False 