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
from config.constants import (
    HEURISTIC_BASE_SCORE, HEURISTIC_WEIGHTS,
    AI_SCORE_MIN_VALID, AI_SCORE_MAX, AI_SCORE_SKIPPED,
    RATIONALE_PATTERNS, TRIVIAL_PATTERNS,
    INPUT_TRUNCATION_LENGTH, MAX_INSTRUCTION_LENGTH, MAX_DIFF_LENGTH
)

logger = logging.getLogger(__name__)

# Compiled regex patterns
RATIONALE_PATTERN = re.compile(r'\b(' + '|'.join(RATIONALE_PATTERNS) + r')\b', re.IGNORECASE)
TRIVIAL_PATTERN = re.compile(r'\b(' + '|'.join(TRIVIAL_PATTERNS) + r')\b', re.IGNORECASE)

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
        
        Scoring:
        - Base: 20 points for any non-trivial commit
        - Metadata: Signed-off (+15), Reviewed (+20), Acked (+10), Fixes (+20)
        - Message: Long (+10), Has rationale (+15)
        - Size: Optimal 1-5 files (+10), Too many >20 (-30)
        - Penalties: Trivial subject (-50)
        """
        # Start with base score for non-trivial commits
        score = HEURISTIC_BASE_SCORE
        
        # 1. Metadata presence
        if signed_off_by:
            score += HEURISTIC_WEIGHTS['signed_off']
        if reviewed_by:
            score += HEURISTIC_WEIGHTS['reviewed_by']
        if acked_by:
            score += HEURISTIC_WEIGHTS['acked_by']
        if fixes_hash:
            score += HEURISTIC_WEIGHTS['fixes_tag']
            
        # 2. Message quality
        full_message = f"{subject}\n{body}"
        if len(full_message) > 200:
            score += HEURISTIC_WEIGHTS['long_message']
            
        if RATIONALE_PATTERN.search(body):
            score += HEURISTIC_WEIGHTS['has_rationale']
            
        if TRIVIAL_PATTERN.search(subject):
            score += HEURISTIC_WEIGHTS['trivial_subject']
            
        # 3. Change size
        if 1 <= files_changed <= 5:
            score += HEURISTIC_WEIGHTS['optimal_size']
        elif files_changed > 20:
            score += HEURISTIC_WEIGHTS['too_many_files']
            
        # Clamp score to 0-100
        return max(0.0, min(100.0, score))

    def get_ai_score(self, instruction: str, diff: str, code_context: str = "") -> tuple[Optional[float], Optional[str]]:
        """
        Get 1-5 quality score from LLM with reasoning.
        
        Args:
            instruction: Commit message
            diff: Code diff
            code_context: Optional code context (smart extracted)
            
        Returns: (score, reasoning) or (None, None) on error.
        """
        # Build prompt with optional context
        context_section = ""
        if code_context:
            context_section = f"""
**Code Context (before change):**
{code_context[:INPUT_TRUNCATION_LENGTH]}
"""
        
        prompt = f"""You are an expert code reviewer evaluating Linux kernel commits for training a code generation AI.

Rate this commit's quality for AI training on a scale of 1-5:

**Scoring Criteria:**
- **5 (Excellent)**: Clear problem statement, detailed explanation of WHY, shows best practices, educational value
- **4 (Good)**: Clear message, logical fix, explains rationale, minor gaps in explanation
- **3 (Average)**: Understandable change, basic explanation, some educational value
- **2 (Below Average)**: Unclear message or trivial change, limited learning value
- **1 (Poor)**: No explanation, confusing, or purely cosmetic change

**Commit Message:**
{instruction[:MAX_INSTRUCTION_LENGTH]}
{context_section}
**Code Diff:**
{diff[:MAX_DIFF_LENGTH]}

**Evaluate:**
1. Does the commit message clearly explain WHAT changed and WHY?
2. Is the code change logical and well-structured?
3. Does it demonstrate good coding practices?
4. Would this help an AI learn to write better code?

Respond ONLY with valid JSON:
{{"score": <1-5>, "reason": "<one detailed sentence explaining your score>"}}
"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # More consistent scoring
                        "top_p": 0.9
                    }
                },
                timeout=90  # Increased timeout for longer context
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract from response
            text = result.get('response', '').strip()
            
            # Try to parse as JSON
            try:
                # Find JSON in response
                json_match = re.search(r'\{[^}]+\}', text)
                if json_match:
                    data = json.loads(json_match.group())
                    score = float(data.get('score', 0))
                    reason = data.get('reason', '')
                    if 1 <= score <= 5:
                        return score, reason
            except json.JSONDecodeError:
                pass
            
            # Fallback: extract just the number
            match = re.search(r'([1-5])', text)
            if match:
                return float(match.group(1)), text[:200]
            
        except Exception as e:
            logger.warning(f"AI scoring failed: {e}")
            
        return None, None

    def get_ai_score_simple(self, instruction: str, diff: str) -> Optional[float]:
        """Legacy method for backward compatibility."""
        score, _ = self.get_ai_score(instruction, diff)
        return score

