"""
JSONL exporter for training data in instruction tuning format.

Exports data from ClickHouse to JSONL format compatible with
CommitPack/OctoPack for LLM fine-tuning.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Iterator, Literal
from dataclasses import dataclass

from clickhouse_driver import Client
from tqdm import tqdm

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example in instruction tuning format."""
    instruction: str
    input: str
    output: str
    
    # Optional metadata for filtering
    commit_hash: Optional[str] = None
    subsystem: Optional[str] = None
    lines_changed: Optional[int] = None


class TrainingDataExporter:
    """
    Exports kernel history data to JSONL format for LLM training.
    
    Supports:
    - Instruction tuning format (instruction/input/output)
    - CommitPack/OctoPack compatible output
    - Diff-based or full-code output modes
    - Subsystem and complexity filtering
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None
    ):
        """
        Initialize the exporter.
        
        Args:
            host: ClickHouse server host.
            port: ClickHouse native port.
            database: Database name.
        """
        self.host = host or settings.clickhouse.host
        self.port = port or settings.clickhouse.port
        self.database = database or settings.clickhouse.database
        self._client: Optional[Client] = None
    
    @property
    def client(self) -> Client:
        """Get or create ClickHouse client."""
        if self._client is None:
            self._client = Client(
                host=self.host,
                port=self.port,
                database=self.database
            )
        return self._client
    
    def _build_query(
        self,
        output_mode: Literal["diff", "full"] = "diff",
        min_lines: int = 5,
        max_lines: int = 500,
        exclude_merges: bool = True,
        subsystems: Optional[list[str]] = None,
        exclude_subsystems: Optional[list[str]] = None,
        limit: Optional[int] = None,
        sample_ratio: float = 1.0,
        min_heuristic_score: float = 0.0,
        min_ai_score: float = 0.0,
        tags: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None
    ) -> str:
        """
        Build the SQL query for extracting training data.
        
        Args:
            output_mode: 'diff' for unified diff, 'full' for complete code.
            min_lines: Minimum lines changed to include.
            max_lines: Maximum lines changed to include.
            exclude_merges: Exclude merge commits.
            subsystems: List of subsystems to include (None = all).
            exclude_subsystems: Subsystems to exclude.
            limit: Maximum number of examples.
            sample_ratio: Random sampling ratio (0-1).
            
        Returns:
            SQL query string.
        """
        # Select fields based on output mode
        if output_mode == "diff":
            output_field = "fc.diff_hunk"
        else:
            output_field = "fc.code_after"
        
        query = f"""
        SELECT
            c.commit_hash,
            c.subject || '\n\n' || c.body AS instruction,
            fc.code_before AS input,
            {output_field} AS output,
            fc.subsystem,
            fc.lines_added + fc.lines_deleted AS lines_changed
        FROM file_changes fc
        INNER JOIN commits c ON fc.commit_hash = c.commit_hash
        LEFT JOIN commit_tags ct ON c.commit_hash = ct.commit_hash
        WHERE 1=1
            AND fc.is_binary = 0
            AND length(fc.diff_hunk) > 10
            AND fc.lines_added + fc.lines_deleted >= {min_lines}
            AND fc.lines_added + fc.lines_deleted <= {max_lines}
        """
        
        if exclude_merges:
            query += "\n            AND c.is_merge = 0"
        
        if subsystems:
            subsystem_list = ', '.join(f"'{s}'" for s in subsystems)
            query += f"\n            AND fc.subsystem IN ({subsystem_list})"
        
        if exclude_subsystems:
            for subsys in exclude_subsystems:
                query += f"\n            AND fc.subsystem != '{subsys}'"
        
        if sample_ratio < 1.0:
            query += f"\n            AND rand() < {sample_ratio}"
        
        if min_heuristic_score > 0:
            query += f"\n            AND c.heuristic_score >= {min_heuristic_score}"
            
        if min_ai_score > 0:
            query += f"\n            AND c.ai_quality_score >= {min_ai_score}"

        if tags:
            for tag in tags:
                query += f"\n            AND has(ct.tags, '{tag}')"
                
        if exclude_tags:
            for tag in exclude_tags:
                query += f"\n            AND NOT has(ct.tags, '{tag}')"
        
        query += "\n        ORDER BY c.commit_date"
        
        if limit:
            query += f"\n        LIMIT {limit}"
        
        return query
    
    def iter_examples(
        self,
        output_mode: Literal["diff", "full"] = "diff",
        min_lines: int = 5,
        max_lines: int = 500,
        exclude_merges: bool = True,
        subsystems: Optional[list[str]] = None,
        exclude_subsystems: Optional[list[str]] = None,
        limit: Optional[int] = None,
        sample_ratio: float = 1.0,
        min_heuristic_score: float = 0.0,
        min_ai_score: float = 0.0,
        batch_size: int = 10000,
        tags: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None
    ) -> Iterator[TrainingExample]:
        """
        Iterate over training examples from the database.
        
        Yields:
            TrainingExample objects.
        """
        query = self._build_query(
            output_mode=output_mode,
            min_lines=min_lines,
            max_lines=max_lines,
            exclude_merges=exclude_merges,
            subsystems=subsystems,
            exclude_subsystems=exclude_subsystems,
            limit=limit,
            sample_ratio=sample_ratio,
            min_heuristic_score=min_heuristic_score,
            min_ai_score=min_ai_score,
            tags=tags,
            exclude_tags=exclude_tags
        )
        
        logger.debug(f"Executing query: {query[:200]}...")
        
        # Stream results
        result = self.client.execute_iter(
            query,
            settings={'max_block_size': batch_size}
        )
        
        for row in result:
            commit_hash, instruction, input_text, output, subsystem, lines_changed = row
            
            # Skip empty or invalid examples
            if not instruction or not output:
                continue
            
            # Clean up the instruction
            instruction = instruction.strip()
            if not instruction:
                continue
            
            yield TrainingExample(
                instruction=instruction,
                input=input_text or "",
                output=output,
                commit_hash=commit_hash,
                subsystem=subsystem,
                lines_changed=lines_changed
            )
    
    def export_jsonl(
        self,
        output_path: Path,
        output_mode: Literal["diff", "full"] = "diff",
        min_lines: int = 5,
        max_lines: int = 500,
        exclude_merges: bool = True,
        subsystems: Optional[list[str]] = None,
        exclude_subsystems: Optional[list[str]] = None,
        limit: Optional[int] = None,
        sample_ratio: float = 1.0,
        include_metadata: bool = False,
        min_heuristic_score: float = 0.0,
        min_ai_score: float = 0.0,
        tags: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None
    ) -> int:
        """
        Export training data to a JSONL file.
        
        Args:
            output_path: Path to the output JSONL file.
            output_mode: 'diff' for unified diff, 'full' for complete code.
            min_lines: Minimum lines changed.
            max_lines: Maximum lines changed.
            exclude_merges: Exclude merge commits.
            subsystems: Subsystems to include.
            exclude_subsystems: Subsystems to exclude.
            limit: Maximum examples to export.
            sample_ratio: Random sampling ratio.
            include_metadata: Include commit hash and subsystem in output.
            
        Returns:
            Number of examples exported.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            examples = self.iter_examples(
                output_mode=output_mode,
                min_lines=min_lines,
                max_lines=max_lines,
                exclude_merges=exclude_merges,
                subsystems=subsystems,
                exclude_subsystems=exclude_subsystems,
                limit=limit,
                sample_ratio=sample_ratio,
                min_heuristic_score=min_heuristic_score,
                min_ai_score=min_ai_score,
                tags=tags,
                exclude_tags=exclude_tags
            )
            
            for example in tqdm(examples, desc="Exporting to JSONL"):
                record = {
                    "instruction": example.instruction,
                    "input": example.input,
                    "output": example.output
                }
                
                if include_metadata:
                    record["_commit_hash"] = example.commit_hash
                    record["_subsystem"] = example.subsystem
                    record["_lines_changed"] = example.lines_changed
                
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
        
        logger.info(f"Exported {count} examples to {output_path}")
        return count
    
    def export_subsystem_datasets(
        self,
        output_dir: Path,
        subsystems: list[str],
        **kwargs
    ) -> dict[str, int]:
        """
        Export separate datasets for each subsystem.
        
        Args:
            output_dir: Directory to save datasets.
            subsystems: List of subsystems to export.
            **kwargs: Additional arguments for export_jsonl.
            
        Returns:
            Dict mapping subsystem names to export counts.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for subsystem in subsystems:
            output_path = output_dir / f"{subsystem.replace('/', '_')}_training.jsonl"
            count = self.export_jsonl(
                output_path=output_path,
                subsystems=[subsystem],
                **kwargs
            )
            results[subsystem] = count
        
        return results
    
    def get_statistics(self) -> dict:
        """Get statistics about the available training data."""
        stats = {}
        
        # Total counts
        result = self.client.execute(
            "SELECT count() FROM commits WHERE is_merge = 0"
        )
        stats['total_commits'] = result[0][0]
        
        result = self.client.execute(
            "SELECT count() FROM file_changes WHERE is_binary = 0"
        )
        stats['total_file_changes'] = result[0][0]
        
        # By subsystem
        result = self.client.execute("""
            SELECT subsystem, count() as cnt
            FROM file_changes
            WHERE is_binary = 0
            GROUP BY subsystem
            ORDER BY cnt DESC
            LIMIT 20
        """)
        stats['top_subsystems'] = {row[0]: row[1] for row in result}
        
        # Lines changed distribution
        result = self.client.execute("""
            SELECT 
                multiIf(
                    lines_added + lines_deleted < 10, '0-10',
                    lines_added + lines_deleted < 50, '10-50',
                    lines_added + lines_deleted < 100, '50-100',
                    lines_added + lines_deleted < 500, '100-500',
                    '500+'
                ) as bucket,
                count() as cnt
            FROM file_changes
            WHERE is_binary = 0
            GROUP BY bucket
            ORDER BY bucket
        """)
        stats['lines_changed_distribution'] = {row[0]: row[1] for row in result}
        
        return stats
    
    def close(self) -> None:
        """Close the database connection."""
        if self._client:
            self._client.disconnect()
            self._client = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
