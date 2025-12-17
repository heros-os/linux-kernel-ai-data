#!/usr/bin/env python3
"""
Synthetic Data Augmentation Script (CLEAR Pipeline)

Identifies "Hidden Gem" commits (good code, poor message) and uses AI to 
rewrite the commit messages into high-quality training examples.

This implements the 'Rectification' layer of the Continuous Adaptation Stack.
"""

import sys
import json
import logging
import requests
import re
from pathlib import Path
from rich.console import Console
from rich.progress import Progress

sys.path.insert(0, str(Path(__file__).parent.parent))

from clickhouse_driver import Client
from config.settings import settings
from config.constants import (
    INPUT_TRUNCATION_LENGTH, 
    MAX_DIFF_LENGTH,
    MAX_INSTRUCTION_LENGTH
)
from src.context_extractor import SmartContextExtractor
from src.validator import TrainingExampleValidator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
console = Console()

class SyntheticDataGenerator:
    def __init__(self):
        self.client = Client(
            host=settings.clickhouse.host,
            port=settings.clickhouse.port,
            database=settings.clickhouse.database
        )
        self.extractor = SmartContextExtractor()
        self.validator = TrainingExampleValidator()
        self.ollama_url = settings.quality.ollama_url
        self.model_name = settings.quality.model_name  # e.g., qwen2.5-coder:7b

    def find_hidden_gems(self, limit: int = 100):
        """
        Find commits with good code characteristics but likely poor messages.
        Criteria:
        - Small, focused change (1-5 files)
        - Non-trivial diff length (100-8000 chars)
        - Short message OR low heuristic score
        """
        query = """
        SELECT 
            c.commit_hash,
            c.subject,
            c.body,
            c.files_changed
        FROM commits c
        WHERE c.files_changed BETWEEN 1 AND 5
          AND length(c.body) < 150  -- Short message (likely poor quality)
          AND (c.ai_quality_score < 4 OR c.ai_quality_score IS NULL) -- Not already scored high by AI
        ORDER BY rand()
        LIMIT %(limit)s
        """
        return self.client.execute(query, {'limit': limit})
    
    def get_diff(self, commit_hash: str) -> str:
        """Get the main diff for the commit."""
        query = """
        SELECT diff_hunk FROM file_changes 
        WHERE commit_hash = %(hash)s 
        ORDER BY length(diff_hunk) DESC 
        LIMIT 1
        """
        result = self.client.execute(query, {'hash': commit_hash})
        return result[0][0] if result else ""

    def generate_instruction(self, original_msg: str, diff: str, context: str) -> str:
        """Use LLM to rewrite the commit message into a gold-standard instruction."""
        prompt = f"""You are a senior Linux kernel maintainer.
The following commit contains a valid code fix but has a poor or lazy commit message.

Your task is to REWRITE the commit message to meet strict Linux kernel standards.
Explain WHAT changed and WHY, considering the code context and diff.

**Standards:**
- Subject line: "subsystem: short summary" (max 75 chars)
- Body: Detailed explanation of the problem and the fix.
- Do NOT make up facts. Stick to what is visible in the code.

**Original Message (for reference):**
{original_msg}

**Code Context:**
{context[:2000]}

**Code Diff:**
{diff[:4000]}

Respond ONLY with the rewritten commit message (Subject + Body). Do not include "Here is the rewritten message" or other filler."""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2, # Low temp for factual accuracy
                        "num_predict": 512
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()['response'].strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

    def run(self, output_path: str, count: int = 50):
        console.print(f"[bold cyan]Searching for 'Hidden Gems' (limit {count})...[/bold cyan]")
        candidates = self.find_hidden_gems(limit=count * 2) # Fetch more to account for extraction failures
        
        console.print(f"[bold green]Found {len(candidates)} candidates. Processing...[/bold green]")
        
        valid_examples = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            with Progress() as progress:
                task = progress.add_task("[cyan]Synthesizing Gold Data...", total=count)
                
                for commit in candidates:
                    if valid_examples >= count:
                        break
                        
                    commit_hash, subject, body, _ = commit
                    original_msg = f"{subject}\n{body}".strip()
                    
                    # 1. Get Diff and Code Context
                    try:
                        # Get first C file diff with code_before
                        diff_query = """
                        SELECT diff_hunk, code_before 
                        FROM file_changes 
                        WHERE commit_hash = %(hash)s 
                          AND (file_extension = '.c' OR file_extension = '.h')
                        ORDER BY length(diff_hunk) DESC 
                        LIMIT 1
                        """
                        diff_res = self.client.execute(diff_query, {'hash': commit_hash})
                        
                        if not diff_res:
                            continue
                            
                        diff = diff_res[0][0]
                        code_before = diff_res[0][1]
                        
                        if not diff or len(diff) < 100 or len(diff) > MAX_DIFF_LENGTH:
                            continue
                            
                        # 2. Extract Smart Context
                        input_context = self.extractor.extract_changed_region(code_before, diff)
                        
                        if not input_context:
                            continue
                            
                    except Exception as e:
                        # logger.warning(f"Context extraction failed for {commit_hash}: {e}")
                        continue

                    # 3. Generate Synthetic Instruction
                    new_instruction = self.generate_instruction(original_msg, diff, input_context)
                    if not new_instruction or len(new_instruction) < 50:
                        continue

                    # 4. Create Record
                    record = {
                        "instruction": new_instruction,
                        "input": self.validator.clean_input(input_context),
                        "output": self.validator.clean_output(diff),
                        "original_instruction": original_msg,
                        "is_synthetic": True,
                        "quality_score": 5.0, # Synthetic Gold is assumed 5
                        "quality_reason": "Synthetic Gold (AI-Rewritten)"
                    }
                    
                    # 5. Validate Final Record
                    is_valid, error = self.validator.is_valid(
                        record['instruction'],
                        record['input'],
                        record['output']
                    )
                    
                    if is_valid:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        valid_examples += 1
                        progress.update(task, advance=1)
                        progress.console.print(f"[dim]Generated: {commit_hash[:8]} -> {len(new_instruction)} chars[/dim]")
                    else:
                        progress.console.print(f"[dim yellow]Validation failed for {commit_hash[:8]}: {error}[/dim yellow]")
                    
        console.print(f"\n[bold green]Generated {valid_examples} synthetic examples to {output_path}[/bold green]")

if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    output_file = "huggingface_datasets/synthetic_gold.jsonl"
    generator.run(output_file, count=500) # User requested 500
