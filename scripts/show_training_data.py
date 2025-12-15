#!/usr/bin/env python3
"""Show what's actually in the exported training data."""
import json
from pathlib import Path

# Check premium_ai_train.jsonl
file_path = Path("huggingface_dataset/premium_ai_train.jsonl")

if not file_path.exists():
    print("File not found. Run: python scripts/prepare_huggingface.py")
    exit(1)

with open(file_path, 'r', encoding='utf-8') as f:
    record = json.loads(f.readline())

print("=" * 70)
print(" ACTUAL TRAINING DATA CONTENT")
print("=" * 70)

print(f"\n[FIELD SIZES]")
print(f"  System prompt: {len(record.get('system', ''))} chars")
print(f"  Instruction:   {len(record['instruction'])} chars")
print(f"  Input:         {len(record['input'])} chars")
print(f"  Output:        {len(record['output'])} chars")

if '_quality_score' in record:
    print(f"  Quality score: {record['_quality_score']}")
    print(f"  Quality reason: {len(record.get('_quality_reason', ''))} chars")

print(f"\n[SYSTEM PROMPT - Full]")
print(record.get('system', 'N/A'))

print(f"\n[INSTRUCTION - First 800 chars]")
print(record['instruction'][:800])
if len(record['instruction']) > 800:
    print(f"... ({len(record['instruction']) - 800} more chars)")

print(f"\n[INPUT - First 800 chars]")
print(record['input'][:800] if record['input'] else "(empty)")
if len(record['input']) > 800:
    print(f"... ({len(record['input']) - 800} more chars)")

print(f"\n[OUTPUT - First 800 chars]")
print(record['output'][:800])
if len(record['output']) > 800:
    print(f"... ({len(record['output']) - 800} more chars)")

if '_quality_reason' in record:
    print(f"\n[AI QUALITY REASONING]")
    print(record['_quality_reason'])

print("\n" + "=" * 70)
print(" SUMMARY")
print("=" * 70)
print(f"Total size: {len(json.dumps(record))} chars")
print(f"Is truncated: {'Yes' if len(record['input']) >= 7999 or len(record['output']) >= 11999 else 'No'}")
