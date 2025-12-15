#!/usr/bin/env python3
"""Validate JSONL file format."""
import json
import sys
from pathlib import Path

def validate_jsonl(file_path):
    """Validate JSONL file and report errors."""
    print(f"Validating: {file_path}")
    print("=" * 70)
    
    if not Path(file_path).exists():
        print(f"ERROR: File not found: {file_path}")
        return False
    
    errors = []
    valid_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                record = json.loads(line)
                valid_count += 1
                
                # Check required fields
                required = ['instruction', 'output']
                missing = [f for f in required if f not in record]
                if missing:
                    errors.append((line_num, f"Missing fields: {missing}"))
                    
            except json.JSONDecodeError as e:
                errors.append((line_num, f"JSON error: {e}"))
                # Show the problematic line
                print(f"\nLine {line_num} (INVALID):")
                print(line[:200] + "..." if len(line) > 200 else line)
                print(f"Error: {e}\n")
    
    print(f"\nResults:")
    print(f"  Valid records: {valid_count}")
    print(f"  Errors: {len(errors)}")
    
    if errors:
        print(f"\nFirst 5 errors:")
        for line_num, error in errors[:5]:
            print(f"  Line {line_num}: {error}")
        return False
    else:
        print("\n[OK] All records valid!")
        
        # Show sample
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = json.loads(f.readline())
            print(f"\nSample record fields:")
            for key in sample.keys():
                value_len = len(str(sample[key]))
                print(f"  - {key}: {value_len} chars")
        
        return True

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "test_smart_context/premium.jsonl"
    validate_jsonl(file_path)
