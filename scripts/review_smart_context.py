#!/usr/bin/env python3
"""Review exported training data with smart context."""
import json
from pathlib import Path

file_path = Path("test_smart_context/premium.jsonl")

with open(file_path, 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f]

print("=" * 80)
print(f" REVIEWING {len(records)} EXPORTED COMMITS WITH SMART CONTEXT")
print("=" * 80)

for i, record in enumerate(records, 1):
    print(f"\n{'='*80}")
    print(f" COMMIT {i}/{len(records)}")
    print(f"{'='*80}")
    
    print(f"\n[INSTRUCTION - Subject]")
    subject = record['instruction'].split('\n')[0]
    print(f"  {subject}")
    
    print(f"\n[INPUT - Smart Context ({len(record['input'])} chars)]")
    print("-" * 40)
    print(record['input'][:600])
    if len(record['input']) > 600:
        print(f"... ({len(record['input']) - 600} more chars)")
    print("-" * 40)
    
    print(f"\n[OUTPUT - Diff ({len(record['output'])} chars)]")
    print("-" * 40)
    print(record['output'][:400])
    if len(record['output']) > 400:
        print(f"... ({len(record['output']) - 400} more chars)")
    print("-" * 40)
    
    # Check if input looks relevant
    has_function = 'static' in record['input'] or 'int ' in record['input'] or 'void ' in record['input']
    has_braces = '{' in record['input'] and '}' in record['input']
    
    print(f"\n[CONTEXT QUALITY]")
    print(f"  Contains function: {'Yes' if has_function else 'No'}")
    print(f"  Contains braces: {'Yes' if has_braces else 'No'}")
    print(f"  Looks relevant: {'Yes' if (has_function and has_braces) else 'Maybe'}")

print("\n" + "=" * 80)
print(" SUMMARY")
print("=" * 80)
print(f"Total records: {len(records)}")
print(f"Avg input size: {sum(len(r['input']) for r in records) / len(records):.0f} chars")
print(f"Avg output size: {sum(len(r['output']) for r in records) / len(records):.0f} chars")
