"""
Quality scoring module for Linux kernel commits.

Implements heuristic and AI-based scoring to filter high-quality training data.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, List
import requests
import json

from config.settings import settings

logger = logging.getLogger(__name__)

# Heuristic scoring weights
WEIGHTS = {
    'signed_off': 15.0,
    'reviewed_by': 20.0,
    'acked_by': 10.0,
    'fixes_tag': 20.0,
    'long_message': 10.0,      # > 200 chars
    'has_rationale': 15.0,     # "because", "since", etc.
    'optimal_size': 10.0,      # 1-5 files
    'too_many_files': -30.0,   # > 20 files
    'trivial_subject': -50.0   # "typo", "cleanup", etc.
}

# Regex for heuristic checks
RATIONALE_PATTERN = re.compile(r'\b(because|since|due to|reason|trigger|cause)\b', re.IGNORECASE)
TRIVIAL_PATTERN = re.compile(r'\b(typo|whitespace|formatting|cleanup|unused|indent|style)\b', re.IGNORECASE)

class QualityScorer:
    """
    Evaluates commit quality using heuristics and LLM scoring.
    """
    
    def __init__(self):
        self.ollama_url = settings.quality.ollama_url
        self.model_name = settings.quality.model_name
    
    def calculate_heuristic_score(
        self,
        subject: str,
        body: str,
        files_changed: int,
        signed_off_by: List[str],
        reviewed_by: List[str],
        acked_by: List[str],
        fixes_hash: Optional[str]
    ) -> float:
        """
        Calculate a 0-100 score based on static features.
        """
        score = 0.0
        
        # 1. Metadata presence
        if signed_off_by:
            score += WEIGHTS['signed_off']
        if reviewed_by:
            score += WEIGHTS['reviewed_by']
        if acked_by:
            score += WEIGHTS['acked_by']
        if fixes_hash:
            score += WEIGHTS['fixes_tag']
            
        # 2. Message quality
        full_message = f"{subject}\n{body}"
        if len(full_message) > 200:
            score += WEIGHTS['long_message']
            
        if RATIONALE_PATTERN.search(body):
            score += WEIGHTS['has_rationale']
            
        if TRIVIAL_PATTERN.search(subject):
            score += WEIGHTS['trivial_subject']
            
        # 3. Change size
        if 1 <= files_changed <= 5:
            score += WEIGHTS['optimal_size']
        elif files_changed > 20:
            score += WEIGHTS['too_many_files']
            
        # Clamp score to 0-100
        return max(0.0, min(100.0, score))

    def get_ai_score(self, instruction: str, diff: str) -> Optional[float]:
        """
        Get 1-5 quality score from LLM.
        """
        prompt = f"""Rate the quality of this Linux kernel commit for training an AI to write code.
Score from 1 to 5 based on:
1. Clarity of the instruction (commit message)
2. Value of the code change (is it a logical fix/feature?)
3. Explanation of WHY the change was made

Commit Message:
{instruction}

Diff:
{diff[:2000]}  # Truncated for context window

Output ONLY a single number 1-5.
"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract number from response
            text = result.get('response', '').strip()
            match = re.search(r'([1-5])', text)
            if match:
                return float(match.group(1))
            
        except Exception as e:
            logger.warning(f"AI scoring failed: {e}")
            
        return None
