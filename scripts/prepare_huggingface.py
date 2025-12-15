#!/usr/bin/env python3
"""
Prepare dataset for HuggingFace upload.

Creates train/validation splits and exports in HuggingFace format.
"""

import sys
import json
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from clickhouse_driver import Client
from config.settings import settings
from config.constants import KERNEL_SYSTEM_PROMPT, INPUT_TRUNCATION_LENGTH, OUTPUT_TRUNCATION_LENGTH

# Set seed for reproducibility
random.seed(42)

client = Client(
    host=settings.clickhouse.host,
    port=settings.clickhouse.port,
    database=settings.clickhouse.database
)

def export_split(output_dir: Path, split_name: str, query: str, limit: int, validation_ratio: float = 0.1):
    """Export a dataset split with train/validation."""
    
    print(f"\nExporting {split_name}...")
    
    # Fetch commits
    commits = client.execute(query, {'limit': limit})
    
    if not commits:
        print(f"  No data found for {split_name}")
        return
    
    # Shuffle for random split
    random.shuffle(commits)
    
    # Split into train/val
    val_size = int(len(commits) * validation_ratio)
    train_commits = commits[val_size:]
    val_commits = commits[:val_size]
    
    print(f"  Total: {len(commits)}, Train: {len(train_commits)}, Val: {len(val_commits)}")
    
    # Export train
    train_path = output_dir / f"{split_name}_train.jsonl"
    export_commits(train_commits, train_path, split_name)
    
    # Export validation
    val_path = output_dir / f"{split_name}_val.jsonl"
    export_commits(val_commits, val_path, split_name)
    
    print(f"  OK - Exported to {output_dir}")


def export_commits(commits, output_path: Path, split_name: str):
    """Export commits to JSONL."""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in commits:
            if "ai" in split_name:
                hash, subject, body, ai_score, ai_reason = row
            else:
                hash, subject, body = row
                ai_score, ai_reason = None, None
            
            # Get file change
            fc_query = """
            SELECT code_before, diff_hunk
            FROM file_changes
            WHERE commit_hash = %(hash)s
                AND is_binary = 0
                AND length(diff_hunk) > 50
                AND length(diff_hunk) < 5000
            LIMIT 1
            """
            fc_result = client.execute(fc_query, {'hash': hash})
            
            if not fc_result:
                continue
            
            code_before, diff = fc_result[0]
            
            # Build record
            record = {
                "system": KERNEL_SYSTEM_PROMPT,
                "instruction": f"{subject}\n\n{body}".strip(),
                "input": (code_before or "")[:INPUT_TRUNCATION_LENGTH],
                "output": diff[:OUTPUT_TRUNCATION_LENGTH]
            }
            
            # Add AI metadata if available
            if ai_score and ai_reason:
                record["_quality_score"] = ai_score
                record["_quality_reason"] = ai_reason
            
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    output_dir = Path("huggingface_dataset")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print(" HUGGINGFACE DATASET PREPARATION")
    print("=" * 60)
    
    # Premium tier (heuristic only)
    premium_query = """
    SELECT commit_hash, subject, body
    FROM commits
    WHERE heuristic_score >= 90
    ORDER BY heuristic_score DESC
    LIMIT %(limit)s
    """
    export_split(output_dir, "premium", premium_query, 5000)
    
    # Premium AI (with reasoning)
    premium_ai_query = """
    SELECT commit_hash, subject, body, ai_quality_score, ai_score_reason
    FROM commits
    WHERE ai_quality_score >= 4
        AND length(ai_score_reason) > 10
    ORDER BY ai_quality_score DESC, heuristic_score DESC
    LIMIT %(limit)s
    """
    export_split(output_dir, "premium_ai", premium_ai_query, 2000)
    
    # High quality tier
    hq_query = """
    SELECT commit_hash, subject, body
    FROM commits
    WHERE heuristic_score >= 70
    ORDER BY heuristic_score DESC
    LIMIT %(limit)s
    """
    export_split(output_dir, "high_quality", hq_query, 20000)
    
    print("\n" + "=" * 60)
    print(" EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\nDataset ready in: {output_dir.absolute()}")
    print("\nNext steps:")
    print("1. Review the files in huggingface_dataset/")
    print("2. Create a HuggingFace account at https://huggingface.co")
    print("3. Install: pip install huggingface_hub")
    print("4. Upload: huggingface-cli upload your-username/linux-kernel-code huggingface_dataset/")


if __name__ == "__main__":
    main()
