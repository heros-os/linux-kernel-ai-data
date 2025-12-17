
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

print("Total commits:", client.execute("SELECT count(*) FROM commits")[0][0])
print("Commits with files_changed 1-5:", client.execute("SELECT count(*) FROM commits WHERE files_changed BETWEEN 1 AND 5")[0][0])
print("Commits with short body:", client.execute("SELECT count(*) FROM commits WHERE length(body) < 150")[0][0])
print("Score distribution:", client.execute("SELECT ai_quality_score, count(*) FROM commits GROUP BY ai_quality_score"))
