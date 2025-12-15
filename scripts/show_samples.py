#!/usr/bin/env python3
"""Show sample exported training data."""
import json
from pathlib import Path

def show_samples():
    files = [
        ("exports/premium.jsonl", "Premium (Heuristic only)"),
        ("exports/premium_with_reasoning.jsonl", "Premium + AI Reasoning")
    ]
    
    for filepath, label in files:
        path = Path(filepath)
        if not path.exists():
            print(f"\n{label}: File not found")
            continue
        
        print(f"\n{'='*70}")
        print(f" {label}")
        print(f"{'='*70}")
        
        with open(path, 'r', encoding='utf-8') as f:
            record = json.loads(f.readline())
        
        # System prompt
        if 'system' in record:
            print(f"\n[SYSTEM PROMPT]")
            print(f"  {record['system'][:150]}...")
        
        # Instruction (commit message)
        print(f"\n[INSTRUCTION - Commit Message]")
        instruction = record['instruction']
        subject = instruction.split('\n')[0]
        print(f"  Subject: {subject}")
        body_preview = '\n  '.join(instruction.split('\n')[2:6])
        print(f"  Body:\n  {body_preview}...")
        
        # Input (code context)
        print(f"\n[INPUT - Code Context]")
        input_text = record['input'][:400] if record['input'] else "(empty)"
        print(f"  {input_text}...")
        
        # Output (diff)
        print(f"\n[OUTPUT - Diff/Patch]")
        output = record['output'][:500]
        print(f"  {output}...")
        
        # Reasoning (if present)
        if '_quality_score' in record:
            print(f"\n[AI QUALITY ASSESSMENT]")
            print(f"  Score: {record['_quality_score']}/5")
            print(f"  Reason: {record.get('_quality_reason', 'N/A')}")
        
        print()

if __name__ == "__main__":
    show_samples()
