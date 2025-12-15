"""
Data validation module for training examples.

Ensures all exported training examples meet quality standards.
"""

import re
from typing import Optional, Tuple

from config.constants import (
    MIN_DIFF_LENGTH, MAX_DIFF_LENGTH,
    MIN_INSTRUCTION_LENGTH,
    MERGE_PATTERNS, TRIVIAL_PATTERNS,
    INPUT_TRUNCATION_LENGTH, OUTPUT_TRUNCATION_LENGTH
)


class TrainingExampleValidator:
    """Validates training examples before export."""
    
    @staticmethod
    def is_valid(
        instruction: str,
        input_text: str,
        output: str,
        strict: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a training example.
        
        Args:
            instruction: The commit message / instruction text
            input_text: The code context (before)
            output: The diff / patch output
            strict: If True, apply stricter validation rules
            
        Returns:
            (is_valid, reason) - reason is None if valid, otherwise error message
        """
        # Check instruction length
        if not instruction or len(instruction.strip()) < MIN_INSTRUCTION_LENGTH:
            return False, f"Instruction too short ({len(instruction)} < {MIN_INSTRUCTION_LENGTH})"
        
        # Check output length
        if not output or len(output.strip()) < MIN_DIFF_LENGTH:
            return False, f"Output too short ({len(output)} < {MIN_DIFF_LENGTH})"
        
        if len(output) > MAX_DIFF_LENGTH:
            return False, f"Output too long ({len(output)} > {MAX_DIFF_LENGTH})"
        
        # Check input has content (important for smart extraction)
        if not input_text or len(input_text.strip()) < 50:
            return False, f"Input too short or empty ({len(input_text)} chars)"
        
        # Check for merge commits
        for pattern in MERGE_PATTERNS:
            if pattern.lower() in instruction.lower():
                return False, f"Merge commit detected: '{pattern}'"
        
        # Strict mode: additional checks
        if strict:
            # Check instruction has body (newline)
            if '\n' not in instruction:
                return False, "Instruction has no body (single line)"
            
            # Check output is actual diff
            if not TrainingExampleValidator._is_valid_diff(output):
                return False, "Output is not a valid diff format"
            
            # Check for trivial changes
            first_line = instruction.split('\n')[0].lower()
            for pattern in TRIVIAL_PATTERNS:
                if pattern in first_line:
                    return False, f"Trivial change detected: '{pattern}'"
        
        return True, None
    
    @staticmethod
    def _is_valid_diff(output: str) -> bool:
        """Check if output looks like a valid diff."""
        # Should contain diff markers
        has_diff_header = 'diff --git' in output or output.startswith('@@')
        has_hunks = '@@ ' in output and '@@' in output
        has_changes = ('+' in output or '-' in output)
        
        return (has_diff_header or has_hunks) and has_changes
    
    @staticmethod
    def clean_instruction(instruction: str) -> str:
        """Clean and normalize instruction text."""
        # Remove excessive whitespace
        lines = instruction.strip().split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        
        # Remove empty lines at start/end
        while cleaned_lines and not cleaned_lines[0]:
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def clean_input(input_text: str, diff: str = "", max_length: int = INPUT_TRUNCATION_LENGTH) -> str:
        """
        Clean and extract relevant input context.
        
        If diff is provided, extracts the changed region.
        Otherwise, truncates intelligently.
        """
        if not input_text:
            return ""
        
        # Try smart extraction if we have a diff
        if diff:
            from src.context_extractor import SmartContextExtractor
            extractor = SmartContextExtractor()
            return extractor.extract_changed_region(input_text, diff, max_length)
        
        # Fallback: intelligent truncation
        text = input_text.strip()
        
        if len(text) <= max_length:
            return text
        
        # Try to truncate at a function boundary
        truncated = text[:max_length]
        
        # Look for last complete function/block
        last_brace = truncated.rfind('\n}')
        if last_brace > max_length * 0.7:  # Keep at least 70%
            truncated = truncated[:last_brace + 2]
        else:
            # Fall back to last newline
            last_newline = truncated.rfind('\n')
            if last_newline > max_length * 0.9:
                truncated = truncated[:last_newline]
        
        return truncated + "\n/* ... truncated ... */"
    
    @staticmethod
    def clean_output(output: str, max_length: int = OUTPUT_TRUNCATION_LENGTH) -> str:
        """Clean and truncate output diff."""
        if not output:
            return ""
        
        text = output.strip()
        
        if len(text) <= max_length:
            return text
        
        # Truncate at hunk boundary
        truncated = text[:max_length]
        
        # Find last complete hunk
        last_hunk = truncated.rfind('\n@@')
        if last_hunk > max_length * 0.7:
            truncated = truncated[:last_hunk]
        
        return truncated


def validate_jsonl_file(filepath: str, strict: bool = False) -> dict:
    """
    Validate all examples in a JSONL file.
    
    Returns:
        Statistics dict with valid/invalid counts and error details
    """
    import json
    from pathlib import Path
    
    stats = {
        'total': 0,
        'valid': 0,
        'invalid': 0,
        'errors': {}
    }
    
    validator = TrainingExampleValidator()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stats['total'] += 1
            
            try:
                record = json.loads(line)
                is_valid, error = validator.is_valid(
                    record.get('instruction', ''),
                    record.get('input', ''),
                    record.get('output', ''),
                    strict=strict
                )
                
                if is_valid:
                    stats['valid'] += 1
                else:
                    stats['invalid'] += 1
                    stats['errors'][line_num] = error
                    
            except json.JSONDecodeError:
                stats['invalid'] += 1
                stats['errors'][line_num] = "Invalid JSON"
    
    return stats
