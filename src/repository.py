"""
Repository management for the Linux Kernel History Extraction Pipeline.
Handles cloning, fetching, and grafting of kernel repositories.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

import pygit2

from config.settings import settings

logger = logging.getLogger(__name__)


class RepositoryManager:
    """
    Manages Git repositories for the extraction pipeline.
    
    Handles:
    - Cloning any Git repository (bare clone for efficiency)
    - Fetching historical repositories for pre-Git history
    - Grafting historical commits to create a unified timeline
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        repo_url: Optional[str] = None,
        repo_dir: Optional[str] = None
    ):
        """
        Initialize the repository manager.
        
        Args:
            data_dir: Directory to store repositories. Defaults to settings.
            repo_url: Git repository URL to clone. Defaults to Linux kernel.
            repo_dir: Local directory name for the repository.
        """
        self.data_dir = data_dir or settings.repository.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Custom repository support
        self.repo_url = repo_url or settings.repository.kernel_url
        
        # Auto-generate directory name from URL if not provided
        if repo_dir:
            self._repo_dir = repo_dir
        elif repo_url:
            # Extract repo name from URL (e.g., "xbmc/xbmc.git" -> "xbmc.git")
            self._repo_dir = repo_url.rstrip('/').split('/')[-1]
            if not self._repo_dir.endswith('.git'):
                self._repo_dir += '.git'
        else:
            self._repo_dir = settings.repository.kernel_dir
        
        self._repo: Optional[pygit2.Repository] = None
    
    @property
    def kernel_path(self) -> Path:
        """Path to the repository (kept for backward compatibility)."""
        return self.data_dir / self._repo_dir
    
    @property
    def repo_path(self) -> Path:
        """Path to the repository."""
        return self.data_dir / self._repo_dir
    
    @property
    def repo(self) -> pygit2.Repository:
        """Get the pygit2 Repository instance, opening if needed."""
        if self._repo is None:
            if not self.kernel_path.exists():
                raise RuntimeError(
                    f"Kernel repository not found at {self.kernel_path}. "
                    "Run clone_kernel() first."
                )
            self._repo = pygit2.Repository(str(self.kernel_path))
        return self._repo
    
    def clone_kernel(self, force: bool = False) -> Path:
        """
        Clone the Git repository.
        
        Uses a bare clone (--bare) for efficiency since we only need
        to read the Git objects, not check out the working directory.
        
        Args:
            force: If True, remove existing repo and re-clone.
            
        Returns:
            Path to the cloned repository.
        """
        repo_path = self.kernel_path
        
        if repo_path.exists():
            if force:
                logger.warning(f"Removing existing repository at {repo_path}")
                import shutil
                shutil.rmtree(repo_path)
            else:
                logger.info(f"Repository already exists at {repo_path}")
                return repo_path
        
        logger.info(f"Cloning repository from {self.repo_url}")
        logger.info("This may take a while depending on repository size...")
        
        # Use subprocess for better progress feedback during clone
        cmd = [
            "git", "clone", "--bare", "--progress",
            self.repo_url,
            str(repo_path)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully cloned to {repo_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            raise
        
        return repo_path
    
    def fetch_historical_repos(self) -> list[Path]:
        """
        Fetch historical repositories for pre-Git kernel history.
        
        These repositories contain reconstructed history from BitKeeper
        and earlier tarball releases (1991-2005).
        
        Returns:
            List of paths to fetched historical repositories.
        """
        fetched = []
        
        for hist_repo in settings.repository.history_repos:
            repo_path = self.data_dir / hist_repo["dir"]
            
            if repo_path.exists():
                logger.info(f"Historical repo {hist_repo['name']} already exists")
                fetched.append(repo_path)
                continue
            
            logger.info(f"Cloning historical repository: {hist_repo['name']}")
            
            cmd = [
                "git", "clone", "--bare", "--progress",
                hist_repo["url"],
                str(repo_path)
            ]
            
            try:
                subprocess.run(cmd, check=True)
                logger.info(f"Successfully cloned {hist_repo['name']}")
                fetched.append(repo_path)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to clone {hist_repo['name']}: {e}")
        
        return fetched
    
    def graft_history(
        self,
        modern_commit: str,
        historical_commit: str,
        historical_repo_path: Path
    ) -> None:
        """
        Graft historical commits onto the modern repository.
        
        This uses `git replace --graft` to connect the modern Git history
        (starting from 2.6.12-rc2 in 2005) to the reconstructed pre-Git
        history, creating a seamless 30+ year timeline.
        
        Args:
            modern_commit: The SHA-1 of the earliest modern commit (parent to replace)
            historical_commit: The SHA-1 of the historical commit to graft as parent
            historical_repo_path: Path to the historical repository
        """
        if not self.kernel_path.exists():
            raise RuntimeError("Kernel repository must be cloned first")
        
        logger.info(f"Grafting {historical_commit[:12]} as parent of {modern_commit[:12]}")
        
        # First, fetch objects from the historical repository
        fetch_cmd = [
            "git", "-C", str(self.kernel_path),
            "fetch", str(historical_repo_path),
            f"+refs/heads/*:refs/remotes/history/*"
        ]
        
        try:
            subprocess.run(fetch_cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to fetch from historical repo: {e}")
            raise
        
        # Now create the graft
        graft_cmd = [
            "git", "-C", str(self.kernel_path),
            "replace", "--graft", modern_commit, historical_commit
        ]
        
        try:
            subprocess.run(graft_cmd, check=True)
            logger.info("Graft created successfully")
        except subprocess.CalledProcessError as e:
            if "already exists" in str(e.stderr if hasattr(e, 'stderr') else ''):
                logger.info("Graft already exists")
            else:
                logger.error(f"Failed to create graft: {e}")
                raise
    
    def get_commit_count(self) -> int:
        """
        Get the total number of commits in the repository.
        
        Returns:
            Number of commits reachable from HEAD.
        """
        count = 0
        for _ in self.repo.walk(self.repo.head.target, pygit2.GIT_SORT_TOPOLOGICAL):
            count += 1
        return count
    
    def get_all_commit_hashes(
        self,
        start_ref: str = "HEAD",
        reverse: bool = True
    ) -> list[str]:
        """
        Get all commit hashes in topological order.
        
        Args:
            start_ref: Reference to start walking from (default: HEAD)
            reverse: If True, return oldest commits first (chronological order)
            
        Returns:
            List of commit SHA-1 hashes.
        """
        try:
            target = self.repo.revparse_single(start_ref).id
        except KeyError:
            logger.error(f"Reference '{start_ref}' not found")
            raise
        
        sort_flags = pygit2.GIT_SORT_TOPOLOGICAL
        if reverse:
            sort_flags |= pygit2.GIT_SORT_REVERSE
        
        hashes = []
        for commit in self.repo.walk(target, sort_flags):
            hashes.append(str(commit.id))
        
        return hashes
    
    def get_commit(self, commit_hash: str) -> pygit2.Commit:
        """
        Load a commit by its hash.
        
        Args:
            commit_hash: The SHA-1 hash of the commit.
            
        Returns:
            The pygit2.Commit object.
        """
        try:
            oid = pygit2.Oid(hex=commit_hash)
            return self.repo.get(oid)
        except (ValueError, KeyError) as e:
            logger.error(f"Commit {commit_hash} not found: {e}")
            raise
    
    def close(self) -> None:
        """Close the repository handle."""
        if self._repo is not None:
            self._repo = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
