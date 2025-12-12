"""Tests for the quality scorer module."""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quality_scorer import QualityScorer, WEIGHTS

class TestQualityScorer:
    """Tests for QualityScorer class."""
    
    def test_heuristic_positive_features(self):
        """Test score accumulation for positive features."""
        scorer = QualityScorer()
        
        score = scorer.calculate_heuristic_score(
            subject="net: core: Fix memory leak in skbuff",
            body="This patch fixes a leak that happens because we forgot to free.\nSigned-off-by: Me",
            files_changed=2,
            signed_off_by=["Me"],
            reviewed_by=["Reviewer"],
            acked_by=[],
            fixes_hash="abc12345"
        )
        
        expected = (
            WEIGHTS['signed_off'] +
            WEIGHTS['reviewed_by'] +
            WEIGHTS['fixes_tag'] +
            WEIGHTS['has_rationale'] +  # "because"
            WEIGHTS['optimal_size']
        )
        
        assert score == expected
        assert 0 <= score <= 100

    def test_heuristic_negative_features(self):
        """Test penalties for negative features."""
        scorer = QualityScorer()
        
        score = scorer.calculate_heuristic_score(
            subject="fix typo",
            body="oops",
            files_changed=50,
            signed_off_by=[],
            reviewed_by=[],
            acked_by=[],
            fixes_hash=None
        )
        
        # Should be floored at 0
        assert score == 0.0
        
    def test_rationale_detection(self):
        """Test detection of rationale keywords."""
        scorer = QualityScorer()
        
        # "because"
        s1 = scorer.calculate_heuristic_score("subj", "it crashed because of null", 1, [], [], [], None)
        assert s1 >= WEIGHTS['has_rationale']
        
        # "due to"
        s2 = scorer.calculate_heuristic_score("subj", "error due to overflow", 1, [], [], [], None)
        assert s2 >= WEIGHTS['has_rationale']
        
        # No rationale
        s3 = scorer.calculate_heuristic_score("subj", "just changing this", 1, [], [], [], None)
        assert s3 < WEIGHTS['has_rationale']

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
