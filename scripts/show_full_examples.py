#!/usr/bin/env python3
"""Show complete training data for AI-scored commits."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clickhouse_driver import Client
from config.settings import settings

client = Client(
    host=settings.clickhouse.host,
    port=settings.clickhouse.port,
    database=settings.clickhouse.database
)

# Get top 5 AI-scored commits
commits_query = """
SELECT commit_hash, subject, body, ai_quality_score, ai_score_reason, heuristic_score
FROM commits 
WHERE ai_score_reason != '' AND length(ai_score_reason) > 10
ORDER BY ai_quality_score DESC
LIMIT 5
"""

commits = client.execute(commits_query)

for i, (hash, subject, body, ai_score, ai_reason, heuristic) in enumerate(commits, 1):
    print("=" * 80)
    print(f" EXAMPLE {i}: {subject[:60]}...")
    print("=" * 80)
    
    # Scores
    print(f"\n[QUALITY SCORES]")
    print(f"  Heuristic Score: {heuristic}/100")
    print(f"  AI Score: {int(ai_score)}/5")
    print(f"  AI Reasoning: {ai_reason}")
    
    # Full instruction (commit message)
    print(f"\n[INSTRUCTION - Full Commit Message]")
    print("-" * 40)
    instruction = f"{subject}\n\n{body}"
    print(instruction[:1500])
    if len(instruction) > 1500:
        print("... (truncated)")
    print("-" * 40)
    
    # Get file change for this commit
    fc_query = """
    SELECT code_before, diff_hunk, file_path, subsystem
    FROM file_changes
    WHERE commit_hash = %(hash)s
        AND is_binary = 0
        AND length(diff_hunk) > 50
    LIMIT 1
    """
    fc_result = client.execute(fc_query, {'hash': hash})
    
    if fc_result:
        code_before, diff, file_path, subsystem = fc_result[0]
        
        print(f"\n[METADATA]")
        print(f"  File: {file_path}")
        print(f"  Subsystem: {subsystem}")
        
        print(f"\n[INPUT - Code Context (before change)]")
        print("-" * 40)
        if code_before:
            print(code_before[:800])
            if len(code_before) > 800:
                print("... (truncated)")
        else:
            print("(no context available)")
        print("-" * 40)
        
        print(f"\n[OUTPUT - Diff/Patch]")
        print("-" * 40)
        print(diff[:1200])
        if len(diff) > 1200:
            print("... (truncated)")
        print("-" * 40)
    
    print("\n\n")
