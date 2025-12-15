#!/usr/bin/env python3
"""
Benchmark script to compare base vs fine-tuned kernel reviewer models.

Runs both models on the same test prompts and scores their responses.
Requires both models to be available in Ollama.

Usage:
    python benchmark_models.py --base qwen2.5-coder:7b --finetuned kernel-reviewer
"""

import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

console = Console()

# Test prompts - real kernel scenarios
TEST_PROMPTS = [
    {
        "id": 1,
        "category": "Bug Fix",
        "instruction": """net/ipv4: Fix use-after-free in tcp_recvmsg

The socket buffer is being accessed after sk_eat_skb() frees it.
This causes a use-after-free when the socket is under heavy load.

Fix by copying the needed data before freeing the buffer.

Fixes: abc123 ("net: optimize tcp receive path")
Cc: stable@vger.kernel.org""",
        "input": """static int tcp_recvmsg(struct sock *sk, struct msghdr *msg, size_t len)
{
    struct sk_buff *skb;
    int copied = 0;
    
    skb = skb_peek(&sk->sk_receive_queue);
    if (!skb)
        return -EAGAIN;
    
    copied = skb_copy_datagram_msg(skb, 0, msg, skb->len);
    sk_eat_skb(sk, skb);
    
    /* BUG: accessing skb after free */
    tcp_rcv_space_adjust(sk, skb->len);
    
    return copied;
}""",
    },
    {
        "id": 2,
        "category": "Memory Leak",
        "instruction": """drm/amd: Fix memory leak in amdgpu_device_init

The firmware buffer allocated in amdgpu_ucode_init_single_fw() is not
freed when the device initialization fails partway through.

Add proper cleanup path to free the firmware buffer on error.

Signed-off-by: Developer Name <dev@example.com>""",
        "input": """int amdgpu_device_init(struct amdgpu_device *adev)
{
    int r;
    
    adev->fw_buf = kmalloc(FW_SIZE, GFP_KERNEL);
    if (!adev->fw_buf)
        return -ENOMEM;
    
    r = amdgpu_ucode_init_single_fw(adev);
    if (r)
        return r;  /* leak: fw_buf not freed */
    
    r = amdgpu_device_ip_init(adev);
    if (r)
        return r;  /* leak: fw_buf not freed */
    
    return 0;
}""",
    },
    {
        "id": 3,
        "category": "Race Condition",
        "instruction": """fs/ext4: Fix race condition in ext4_writepages

There's a race between ext4_writepages() and ext4_da_invalidatepage()
where the page can be invalidated while we're still writing it.

Take the page lock before checking page state to prevent this race.

Reported-by: syzbot+abc123@syzkaller.appspotmail.com""",
        "input": """static int ext4_writepages(struct address_space *mapping,
                          struct writeback_control *wbc)
{
    struct page *page;
    pgoff_t index = wbc->range_start >> PAGE_SHIFT;
    
    page = find_get_page(mapping, index);
    if (!page)
        return 0;
    
    /* Race: page can be invalidated here */
    if (!PageDirty(page)) {
        put_page(page);
        return 0;
    }
    
    return ext4_do_writepage(page, wbc);
}""",
    },
    {
        "id": 4,
        "category": "NULL Pointer",
        "instruction": """usb/core: Add missing NULL check in usb_get_descriptor

usb_get_descriptor() can return NULL if memory allocation fails,
but the caller doesn't check for this before dereferencing.

Add NULL check to prevent kernel oops.

Fixes: def456 ("usb: refactor descriptor handling")""",
        "input": """int usb_get_device_descriptor(struct usb_device *dev)
{
    struct usb_device_descriptor *desc;
    int ret;
    
    desc = usb_get_descriptor(dev, USB_DT_DEVICE, 0);
    /* Missing NULL check */
    
    ret = usb_parse_device_descriptor(dev, desc);
    kfree(desc);
    
    return ret;
}""",
    },
    {
        "id": 5,
        "category": "Locking",
        "instruction": """kernel/sched: Fix missing rcu_read_lock in task_numa_work

task_numa_work() accesses task->mm without holding rcu_read_lock(),
which can lead to accessing freed memory if the mm is freed
concurrently.

Add proper RCU protection around mm access.

Reviewed-by: Scheduler Maintainer <sched@kernel.org>""",
        "input": """void task_numa_work(struct callback_head *work)
{
    struct task_struct *p = current;
    struct mm_struct *mm = p->mm;
    struct vm_area_struct *vma;
    
    /* mm can be freed here without RCU protection */
    if (!mm || !mm->numa_next_scan)
        return;
    
    vma = find_vma(mm, mm->numa_next_scan);
    if (vma)
        do_numa_migration(vma);
}""",
    },
]


@dataclass
class ModelResponse:
    """Response from a model."""
    model: str
    prompt_id: int
    response: str
    time_seconds: float
    error: Optional[str] = None


def query_ollama(model: str, instruction: str, input_text: str, timeout: int = 120) -> tuple[str, float]:
    """Query Ollama model and return response with timing."""
    
    prompt = f"""### Instruction
{instruction}

### Input
{input_text}

### Response
Generate the kernel patch to fix this issue:"""
    
    start = time.time()
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 1024,
                }
            },
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        elapsed = time.time() - start
        return result.get("response", ""), elapsed
    except Exception as e:
        return f"ERROR: {e}", time.time() - start


def score_response(response: str, category: str) -> dict:
    """Score a response on various criteria."""
    scores = {
        "has_diff": 0,
        "diff_format": 0,
        "has_context": 0,
        "addresses_issue": 0,
        "kernel_style": 0,
    }
    
    # Check for diff format
    if "diff --git" in response or "@@" in response:
        scores["has_diff"] = 1
        
        # Check proper diff format
        if "---" in response and "+++" in response:
            scores["diff_format"] = 1
    
    # Check for context lines
    if response.count("\n ") > 2:  # Context lines start with space
        scores["has_context"] = 1
    
    # Check if it addresses the specific issue
    category_keywords = {
        "Bug Fix": ["copy", "before", "free", "fix"],
        "Memory Leak": ["kfree", "free", "goto", "err", "cleanup"],
        "Race Condition": ["lock", "unlock", "spin_lock", "mutex"],
        "NULL Pointer": ["if (!", "NULL", "check", "!desc"],
        "Locking": ["rcu_read_lock", "rcu_read_unlock", "RCU"],
    }
    
    keywords = category_keywords.get(category, [])
    if any(kw.lower() in response.lower() for kw in keywords):
        scores["addresses_issue"] = 1
    
    # Check kernel coding style indicators
    kernel_patterns = ["#include", "static ", "struct ", "return ", "\t"]
    if sum(1 for p in kernel_patterns if p in response) >= 2:
        scores["kernel_style"] = 1
    
    return scores


def run_benchmark(base_model: str, finetuned_model: str, num_tests: int = 5):
    """Run the benchmark comparison."""
    
    console.print("\n" + "=" * 70)
    console.print("[bold blue] MODEL COMPARISON BENCHMARK [/bold blue]")
    console.print("=" * 70)
    console.print(f"\nBase Model: [cyan]{base_model}[/cyan]")
    console.print(f"Fine-tuned: [green]{finetuned_model}[/green]")
    console.print(f"Test Cases: {num_tests}")
    
    results = {"base": [], "finetuned": []}
    
    tests = TEST_PROMPTS[:num_tests]
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Running benchmarks...", total=len(tests) * 2)
        
        for test in tests:
            # Query base model
            progress.update(task, description=f"[cyan]Base model - Test {test['id']}...")
            response, elapsed = query_ollama(base_model, test["instruction"], test["input"])
            scores = score_response(response, test["category"])
            results["base"].append({
                "id": test["id"],
                "category": test["category"],
                "response": response,
                "time": elapsed,
                "scores": scores,
            })
            progress.advance(task)
            
            # Query fine-tuned model
            progress.update(task, description=f"[green]Fine-tuned - Test {test['id']}...")
            response, elapsed = query_ollama(finetuned_model, test["instruction"], test["input"])
            scores = score_response(response, test["category"])
            results["finetuned"].append({
                "id": test["id"],
                "category": test["category"],
                "response": response,
                "time": elapsed,
                "scores": scores,
            })
            progress.advance(task)
    
    return results


def print_results(results: dict, base_model: str, finetuned_model: str):
    """Print benchmark results."""
    
    console.print("\n" + "=" * 70)
    console.print("[bold green] RESULTS [/bold green]")
    console.print("=" * 70)
    
    # Summary table
    table = Table(title="Score Comparison")
    table.add_column("Test", style="cyan")
    table.add_column("Category")
    table.add_column(f"Base ({base_model})", justify="center")
    table.add_column(f"Fine-tuned ({finetuned_model})", justify="center")
    table.add_column("Winner", justify="center")
    
    base_total = 0
    ft_total = 0
    
    for base, ft in zip(results["base"], results["finetuned"]):
        base_score = sum(base["scores"].values())
        ft_score = sum(ft["scores"].values())
        base_total += base_score
        ft_total += ft_score
        
        if ft_score > base_score:
            winner = "[green]Fine-tuned[/green]"
        elif base_score > ft_score:
            winner = "[red]Base[/red]"
        else:
            winner = "[yellow]Tie[/yellow]"
        
        table.add_row(
            f"#{base['id']}",
            base["category"],
            f"{base_score}/5",
            f"{ft_score}/5",
            winner
        )
    
    table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        f"[bold]{base_total}[/bold]",
        f"[bold]{ft_total}[/bold]",
        "[bold green]Fine-tuned[/bold green]" if ft_total > base_total else 
        "[bold red]Base[/bold red]" if base_total > ft_total else "[bold yellow]Tie[/bold yellow]"
    )
    
    console.print(table)
    
    # Timing comparison
    base_time = sum(r["time"] for r in results["base"])
    ft_time = sum(r["time"] for r in results["finetuned"])
    
    console.print(f"\n[dim]Average response time:[/dim]")
    console.print(f"  Base: {base_time/len(results['base']):.2f}s")
    console.print(f"  Fine-tuned: {ft_time/len(results['finetuned']):.2f}s")
    
    # Show example responses
    console.print("\n" + "=" * 70)
    console.print("[bold] EXAMPLE COMPARISON (Test #1) [/bold]")
    console.print("=" * 70)
    
    console.print(Panel(
        results["base"][0]["response"][:800] + "...",
        title=f"[red]Base Model[/red]",
        border_style="red"
    ))
    
    console.print(Panel(
        results["finetuned"][0]["response"][:800] + "...",
        title=f"[green]Fine-tuned Model[/green]",
        border_style="green"
    ))
    
    # Save full results
    output_path = Path("benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[dim]Full results saved to: {output_path}[/dim]")
    
    # Summary
    console.print("\n" + "=" * 70)
    if ft_total > base_total:
        improvement = ((ft_total - base_total) / base_total) * 100 if base_total > 0 else 100
        console.print(f"[bold green]✅ Fine-tuned model is {improvement:.1f}% better![/bold green]")
    elif base_total > ft_total:
        console.print(f"[bold red]❌ Base model performed better (may need more training)[/bold red]")
    else:
        console.print(f"[bold yellow]➡️ Models performed equally[/bold yellow]")
    console.print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark base vs fine-tuned models")
    parser.add_argument("--base", default="qwen2.5-coder:7b",
                       help="Base model name in Ollama")
    parser.add_argument("--finetuned", default="kernel-reviewer",
                       help="Fine-tuned model name in Ollama")
    parser.add_argument("--tests", type=int, default=5,
                       help="Number of test cases to run (max 5)")
    
    args = parser.parse_args()
    
    # Check Ollama is running
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except:
        console.print("[red]Error: Ollama is not running![/red]")
        console.print("Start Ollama first: ollama serve")
        return
    
    # Run benchmark
    results = run_benchmark(args.base, args.finetuned, min(args.tests, 5))
    print_results(results, args.base, args.finetuned)


if __name__ == "__main__":
    main()
