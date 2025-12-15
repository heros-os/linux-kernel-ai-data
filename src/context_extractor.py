"""
Improved smart context extraction for training data.

Key improvements:
1. Better function boundary detection
2. Smarter context window sizing
3. Multi-hunk support
4. Better handling of edge cases
"""

import re
from typing import Optional, Tuple, List


class ImprovedContextExtractor:
    """Extract relevant code context around changes with improved algorithms."""
    
    @staticmethod
    def extract_changed_region(code_before: str, diff: str, max_chars: int = 8000) -> str:
        """
        Extract the code region that was actually changed.
        
        Improvements over original:
        - Better function detection (handles multi-line signatures)
        - Extracts all affected functions for multi-hunk diffs
        - Smarter context window sizing
        - Better fallback strategies
        
        Args:
            code_before: Full file content before change
            diff: Unified diff
            max_chars: Maximum characters to return
            
        Returns:
            Relevant code context around the change
        """
        if not code_before or not diff:
            return code_before[:max_chars] if code_before else ""
        
        # Parse diff to get all changed line ranges
        hunks = ImprovedContextExtractor._parse_diff_hunks(diff)
        
        if not hunks:
            # Fallback to first N chars
            return code_before[:max_chars]
        
        # Split code into lines
        lines = code_before.split('\n')
        
        # Find all functions containing changes
        affected_functions = []
        for hunk in hunks:
            func_range = ImprovedContextExtractor._find_function_range(
                lines, hunk['old_start'], hunk['old_end']
            )
            if func_range:
                affected_functions.append(func_range)
        
        if affected_functions:
            # Merge overlapping function ranges
            merged = ImprovedContextExtractor._merge_ranges(affected_functions)
            
            # Extract all affected functions with context
            extracted_parts = []
            total_len = 0
            
            for start, end in merged:
                # Add some context before/after
                context_start = max(0, start - 3)
                context_end = min(len(lines), end + 3)
                
                part = '\n'.join(lines[context_start:context_end])
                
                # Check if adding this part would exceed max_chars
                if total_len + len(part) + 10 > max_chars:  # +10 for separator
                    if not extracted_parts:
                        # If this is the first part and it's too large, truncate it
                        part = part[:max_chars - total_len]
                        extracted_parts.append(part)
                    break
                
                extracted_parts.append(part)
                total_len += len(part) + 10
            
            result = '\n\n/* ... */\n\n'.join(extracted_parts)
            
            if len(result) <= max_chars:
                return result
        
        # Fallback: extract region around all changed lines
        all_changed_lines = []
        for hunk in hunks:
            all_changed_lines.extend(range(hunk['old_start'], hunk['old_end'] + 1))
        
        if all_changed_lines:
            first_changed = min(all_changed_lines) - 1  # 0-indexed
            last_changed = max(all_changed_lines) - 1
            
            # Extract with generous context
            start = max(0, first_changed - 20)
            end = min(len(lines), last_changed + 20)
            
            extracted = '\n'.join(lines[start:end])
            
            if len(extracted) > max_chars:
                # Truncate but try to keep complete lines
                extracted = extracted[:max_chars]
                last_newline = extracted.rfind('\n')
                if last_newline > max_chars * 0.9:
                    extracted = extracted[:last_newline]
                extracted += "\n/* ... truncated ... */"
            
            return extracted
        
        # Final fallback
        return code_before[:max_chars]
    
    @staticmethod
    def _parse_diff_hunks(diff: str) -> List[dict]:
        """
        Parse diff to extract all hunk line ranges.
        
        Returns list of dicts with old_start, old_end, old_count
        """
        hunks = []
        
        # Find all hunk headers: @@ -old_start,old_count +new_start,new_count @@
        hunk_pattern = r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@'
        
        for match in re.finditer(hunk_pattern, diff):
            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) else 1
            
            hunks.append({
                'old_start': old_start,
                'old_count': old_count,
                'old_end': old_start + old_count - 1
            })
        
        return hunks
    
    @staticmethod
    def _find_function_range(lines: List[str], start_line: int, end_line: int) -> Optional[Tuple[int, int]]:
        """
        Find the function containing the given line range.
        
        Improved to handle:
        - Multi-line function signatures
        - Static/inline modifiers
        - Various C/C++ function patterns
        """
        if not lines or start_line < 1 or start_line > len(lines):
            return None
        
        # Convert to 0-indexed
        start_idx = start_line - 1
        end_idx = min(end_line - 1, len(lines) - 1)
        
        # Search backwards for function start
        function_start = None
        
        # Patterns for function signatures
        func_patterns = [
            # Standard C function: return_type function_name(params)
            r'^(?:static\s+)?(?:inline\s+)?(?:const\s+)?[\w\s\*]+\s+\w+\s*\([^)]*\)\s*{?',
            # Function pointer or complex return type
            r'^(?:static\s+)?(?:inline\s+)?[\w\s\*]+\s*\(\s*\*\s*\w+\s*\)\s*\([^)]*\)\s*{?',
            # Constructor/destructor (C++)
            r'^(?:virtual\s+)?~?\w+\s*\([^)]*\)\s*{?',
            # Operator overload (C++)
            r'^(?:virtual\s+)?[\w\s\*]+\s+operator\s*[\+\-\*\/\=\<\>]+\s*\([^)]*\)\s*{?',
        ]
        
        for i in range(min(start_idx, len(lines) - 1), max(-1, start_idx - 100), -1):
            line = lines[i].strip()
            
            # Check for function patterns
            for pattern in func_patterns:
                if re.match(pattern, line):
                    function_start = i
                    break
            
            if function_start is not None:
                break
            
            # Also check for opening brace on its own line after function signature
            if line == '{' and i > 0:
                prev_line = lines[i-1].strip()
                # Check if previous line looks like function signature
                if re.match(r'.*\w+\s*\([^)]*\)', prev_line):
                    function_start = i - 1
                    break
        
        if function_start is None:
            return None
        
        # Search forward for function end (matching braces)
        brace_count = 0
        function_end = None
        started_counting = False
        
        for i in range(function_start, min(len(lines), function_start + 500)):
            line = lines[i]
            
            # Count braces
            for char in line:
                if char == '{':
                    brace_count += 1
                    started_counting = True
                elif char == '}':
                    brace_count -= 1
            
            # Found matching closing brace
            if started_counting and brace_count == 0:
                function_end = i
                break
        
        if function_end is not None:
            return (function_start, function_end)
        
        # If we can't find the end, return a reasonable range
        return (function_start, min(len(lines) - 1, function_start + 100))
    
    @staticmethod
    def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping or adjacent ranges."""
        if not ranges:
            return []
        
        # Sort by start position
        sorted_ranges = sorted(ranges)
        merged = [sorted_ranges[0]]
        
        for current_start, current_end in sorted_ranges[1:]:
            last_start, last_end = merged[-1]
            
            # If ranges overlap or are adjacent (within 5 lines), merge them
            if current_start <= last_end + 5:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        
        return merged


# For backward compatibility, create alias
SmartContextExtractor = ImprovedContextExtractor
