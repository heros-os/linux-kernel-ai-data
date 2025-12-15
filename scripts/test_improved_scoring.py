#!/usr/bin/env python3
"""Test the improved AI scoring."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clickhouse_driver import Client
from config.settings import settings
from src.quality_scorer import QualityScorer

client = Client(
    host=settings.clickhouse.host,
    port=settings.clickhouse.port,
    database=settings.clickhouse.database
)

scorer = QualityScorer()

# Get a sample commit
query = """
SELECT commit_hash, subject, body
FROM commits
WHERE heuristic_score >= 80
    AND (ai_score_reason IS NULL OR ai_score_reason = '')
LIMIT 1
"""

result = client.execute(query)
if not result:
    print("No commits found to test")
    sys.exit(1)

hash, subject, body = result[0]
instruction = f"{subject}\n\n{body}"

# Get diff
diff_query = """
SELECT diff_hunk FROM file_changes
WHERE commit_hash = %(hash)s
    AND length(diff_hunk) > 100
    AND length(diff_hunk) < 3000
LIMIT 1
"""
diff_result = client.execute(diff_query, {'hash': hash})

if not diff_result:
    print("No diff found")
    sys.exit(1)

diff = diff_result[0][0]

print("=" * 70)
print(" TESTING IMPROVED AI SCORING")
print("=" * 70)
print(f"\nCommit: {hash[:12]}")
print(f"Subject: {subject[:60]}...")
print(f"\nInstruction length: {len(instruction)} chars")
print(f"Diff length: {len(diff)} chars")
print("\n" + "=" * 70)
print(" SCORING...")
print("=" * 70)

score, reason = scorer.get_ai_score(instruction, diff)

if score:
    print(f"\n[RESULT]")
    print(f"  Score: {score}/5")
    print(f"  Reason: {reason}")
    print("\n" + "=" * 70)
    print(" SUCCESS!")
    print("=" * 70)
else:
    print("\n[ERROR] Scoring failed")
