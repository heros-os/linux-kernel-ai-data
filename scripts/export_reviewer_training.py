#!/usr/bin/env python3
"""
Export AI scoring training data for fine-tuning the reviewer model.

This creates a dataset to improve the AI's code review capabilities.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from clickhouse_driver import Client
from config.settings import settings
from config.constants import MAX_INSTRUCTION_LENGTH, MAX_DIFF_LENGTH

client = Client(
    host=settings.clickhouse.host,
    port=settings.clickhouse.port,
    database=settings.clickhouse.database
)

def export_reviewer_training_data(output_path: Path, min_score: float = 4.0):
    """
    Export high-quality AI reviews for fine-tuning.
    
    Format: instruction (commit+diff) -> output (score + reasoning)
    """
    
    # Get commits with good AI scores
    query = """
    SELECT 
        c.commit_hash,
        c.subject,
        c.body,
        c.ai_quality_score,
        c.ai_score_reason
    FROM commits c
    WHERE c.ai_quality_score >= %(min_score)s
        AND length(c.ai_score_reason) > 20
    ORDER BY c.ai_quality_score DESC
    LIMIT 2000
    """
    
    commits = client.execute(query, {'min_score': min_score})
    
    print(f"Found {len(commits)} high-quality reviews to export")
    
    exported = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for hash, subject, body, score, reason in commits:
            # Get diff
            diff_query = """
            SELECT diff_hunk FROM file_changes
            WHERE commit_hash = %(hash)s
                AND length(diff_hunk) > 100
                AND length(diff_hunk) < 8000
            LIMIT 1
            """
            diff_result = client.execute(diff_query, {'hash': hash})
            
            if not diff_result:
                continue
            
            diff = diff_result[0][0]
            instruction = f"{subject}\n\n{body}".strip()
            
            # Create training example for code review
            record = {
                "system": "You are an expert code reviewer evaluating commits for AI training quality.",
                "instruction": f"""Rate this Linux kernel commit for training quality.

**Commit Message:**
{instruction[:MAX_INSTRUCTION_LENGTH]}

**Code Diff:**
{diff[:MAX_DIFF_LENGTH]}

Provide a score (1-5) and detailed reasoning.""",
                "output": f'{{"score": {int(score)}, "reason": "{reason}"}}'
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            exported += 1
    
    print(f"Exported {exported} examples to {output_path}")
    return exported


def main():
    output_dir = Path("reviewer_training")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print(" AI REVIEWER SELF-IMPROVEMENT DATASET")
    print("=" * 60)
    
    # Export high-quality reviews (score >= 4)
    output_path = output_dir / "reviewer_training.jsonl"
    count = export_reviewer_training_data(output_path, min_score=4.0)
    
    size_mb = output_path.stat().st_size / 1_000_000
    
    print("\n" + "=" * 60)
    print(" EXPORT COMPLETE")
    print("=" * 60)
    print(f"\nFile: {output_path}")
    print(f"Examples: {count}")
    print(f"Size: {size_mb:.2f} MB")
    print("\nNext: Use this to fine-tune qwen2.5-coder:7b")
    print("See: REVIEWER_FINETUNING_GUIDE.md")


if __name__ == "__main__":
    main()
