"""
Centralized constants for the training data pipeline.

All configurable parameters should be defined here for consistency.
"""

# Diff size limits
MIN_DIFF_LENGTH = 50
MAX_DIFF_LENGTH = 8000  # Increased from 5000

# Instruction limits
MIN_INSTRUCTION_LENGTH = 50
MAX_INSTRUCTION_LENGTH = 15000

# Input/Output truncation for training data
INPUT_TRUNCATION_LENGTH = 8000  # Increased from 4000
OUTPUT_TRUNCATION_LENGTH = 12000  # Increased from 8000

# Heuristic scoring
HEURISTIC_BASE_SCORE = 20.0  # Base points for any non-trivial commit
HEURISTIC_WEIGHTS = {
    'signed_off': 15.0,       # Has Signed-off-by
    'reviewed_by': 20.0,      # Has Reviewed-by
    'acked_by': 10.0,         # Has Acked-by
    'fixes_tag': 20.0,        # Has Fixes: tag
    'long_message': 10.0,     # > 200 chars
    'has_rationale': 15.0,    # "because", "since", etc.
    'optimal_size': 10.0,     # 1-5 files changed
    'too_many_files': -30.0,  # > 20 files (penalty)
    'trivial_subject': -50.0  # "typo", "cleanup" (penalty)
}

# AI scoring
AI_SCORE_MIN_VALID = 1.0
AI_SCORE_MAX = 5.0
AI_SCORE_SKIPPED = -1.0  # Used to mark commits without valid diffs

# Export settings
DEFAULT_BATCH_SIZE = 100
DEFAULT_LIMIT = 10000

# Quality thresholds
PREMIUM_MIN_SCORE = 90
HIGH_QUALITY_MIN_SCORE = 70
STANDARD_MIN_SCORE = 50

# Training format
DEFAULT_SYSTEM_PROMPT = """You are an expert software developer. Given a commit message describing a change and the relevant code context, generate the appropriate code patch or modification."""

KERNEL_SYSTEM_PROMPT = """You are a Linux kernel developer. Given a commit message describing a bug fix, feature, or improvement, and the relevant source code context, generate the appropriate kernel patch."""

# File extension filters
CODE_EXTENSIONS = ['.c', '.h', '.py', '.js', '.ts', '.go', '.rs', '.java', '.cpp', '.hpp']
KERNEL_EXTENSIONS = ['.c', '.h']

# Validation rules
MERGE_PATTERNS = ['Merge branch', 'Merge pull request', 'Merge commit', 'Merge remote']
TRIVIAL_PATTERNS = ['typo', 'whitespace', 'formatting', 'cleanup', 'unused', 'indent', 'style']
RATIONALE_PATTERNS = ['because', 'since', 'due to', 'reason', 'trigger', 'cause', 'fixes', 'resolves']
