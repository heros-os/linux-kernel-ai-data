import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clickhouse_driver import Client
from config.settings import settings
from collections import Counter

client = Client(host=settings.clickhouse.host, port=settings.clickhouse.port, database=settings.clickhouse.database)

print("Fetching score distribution...")
scores = client.execute("SELECT heuristic_score FROM commits WHERE heuristic_score > 0")
scores = [s[0] for s in scores]

print(f"Total scored: {len(scores)}")
c = Counter(scores)
print("\nTop 20 most common scores:")
for score, count in c.most_common(20):
    print(f"Score {score}: {count} ({count/len(scores)*100:.1f}%)")
