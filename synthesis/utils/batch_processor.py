"""
Batch processing utilities with rate limiting and concurrency control.
"""

import time
from typing import List, Callable, Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock


@dataclass
class BatchResult:
    """Result from batch processing."""
    success: List[Any]
    failed: List[Dict[str, Any]]
    total: int
    success_count: int
    failure_count: int


class RateLimiter:
    """Thread-safe rate limiter using sliding window."""

    def __init__(self, requests_per_minute: int):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.window = 60.0  # 1 minute window
        self.min_interval = self.window / requests_per_minute
        self.request_times = []
        self.lock = Lock()

    def wait(self):
        """Wait if necessary to respect rate limit."""
        with self.lock:
            current_time = time.time()

            # Remove requests older than the window
            cutoff = current_time - self.window
            self.request_times = [t for t in self.request_times if t > cutoff]

            # If we've hit the limit, wait
            if len(self.request_times) >= self.requests_per_minute:
                oldest_time = self.request_times[0]
                wait_time = oldest_time + self.window - current_time
                if wait_time > 0:
                    print(f"[RateLimiter] Hit {self.requests_per_minute} req/min limit. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    current_time = time.time()

            # Also enforce minimum interval between requests
            if self.request_times:
                time_since_last = current_time - self.request_times[-1]
                if time_since_last < self.min_interval:
                    wait_needed = self.min_interval - time_since_last
                    print(f"[RateLimiter] Enforcing {self.min_interval:.1f}s interval. Waiting {wait_needed:.1f}s...")
                    time.sleep(wait_needed)
                    current_time = time.time()

            # Record this request
            self.request_times.append(current_time)
            print(f"[RateLimiter] Request allowed. Total in window: {len(self.request_times)}/{self.requests_per_minute}")


class BatchProcessor:
    """Process items in batches with rate limiting and concurrency control."""

    def __init__(
        self,
        batch_size: int = 10,
        max_workers: int = 5,
        requests_per_minute: int = 20
    ):
        """
        Initialize batch processor.

        Args:
            batch_size: Number of items to process together
            max_workers: Maximum concurrent workers
            requests_per_minute: Rate limit (requests per minute)
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(requests_per_minute)

    def create_batches(self, items: List[Any]) -> List[List[Any]]:
        """
        Split items into batches.

        Args:
            items: List of items to batch

        Returns:
            List of batches
        """
        batches = []
        for i in range(0, len(items), self.batch_size):
            batches.append(items[i:i + self.batch_size])
        return batches

    def process_batch(
        self,
        batch: List[Any],
        processor_func: Callable,
        **kwargs
    ) -> List[Any]:
        """
        Process a single batch.

        Args:
            batch: Batch of items
            processor_func: Function to process the batch
            **kwargs: Additional arguments for processor_func

        Returns:
            Processing results
        """
        self.rate_limiter.wait()
        return processor_func(batch, **kwargs)

    def process_all(
        self,
        items: List[Any],
        processor_func: Callable,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> BatchResult:
        """
        Process all items in batches.

        Args:
            items: All items to process
            processor_func: Function to process each batch
            progress_callback: Optional callback(current, total) for progress
            **kwargs: Additional arguments for processor_func

        Returns:
            BatchResult with success/failure information
        """
        batches = self.create_batches(items)
        total_batches = len(batches)

        success_results = []
        failed_results = []

        for i, batch in enumerate(batches):
            try:
                result = self.process_batch(batch, processor_func, **kwargs)
                success_results.extend(result)

                if progress_callback:
                    progress_callback(i + 1, total_batches)

            except Exception as e:
                print(f"Batch {i + 1}/{total_batches} failed: {e}")
                failed_results.append({
                    'batch_index': i,
                    'items': batch,
                    'error': str(e)
                })

        return BatchResult(
            success=success_results,
            failed=failed_results,
            total=len(items),
            success_count=len(success_results),
            failure_count=len(items) - len(success_results)
        )

    def process_items_individually(
        self,
        items: List[Any],
        processor_func: Callable[[Any], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        stop_on_error: bool = False
    ) -> BatchResult:
        """
        Process items individually with rate limiting and concurrency.

        Args:
            items: Items to process
            processor_func: Function to process single item
            progress_callback: Optional callback(current, total) for progress
            stop_on_error: Stop processing if an error occurs

        Returns:
            BatchResult with success/failure information
        """
        success_results = []
        failed_results = []
        total = len(items)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            for i, item in enumerate(items):
                # Wait for rate limit
                self.rate_limiter.wait()

                # Submit task
                future = executor.submit(processor_func, item)
                futures[future] = (i, item)

            # Collect results
            completed = 0
            for future in as_completed(futures):
                completed += 1
                idx, item = futures[future]

                try:
                    result = future.result()
                    success_results.append(result)

                    if progress_callback:
                        progress_callback(completed, total)

                except Exception as e:
                    print(f"Item {idx + 1}/{total} failed: {e}")
                    failed_results.append({
                        'index': idx,
                        'item': item,
                        'error': str(e)
                    })

                    if stop_on_error:
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break

        return BatchResult(
            success=success_results,
            failed=failed_results,
            total=total,
            success_count=len(success_results),
            failure_count=len(failed_results)
        )


class ProgressTracker:
    """Simple progress tracker for batch operations."""

    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items
            description: Description of the operation
        """
        self.total = total
        self.description = description
        self.start_time = time.time()

    def update(self, current: int):
        """
        Update progress.

        Args:
            current: Current progress count
        """
        elapsed = time.time() - self.start_time
        percent = (current / self.total) * 100 if self.total > 0 else 0

        # Estimate time remaining
        if current > 0:
            rate = current / elapsed
            remaining = (self.total - current) / rate
            eta_str = f", ETA: {remaining:.0f}s"
        else:
            eta_str = ""

        print(f"\r{self.description}: {current}/{self.total} ({percent:.1f}%){eta_str}", end="", flush=True)

        if current >= self.total:
            print()  # New line when complete

    def __call__(self, current: int, total: int):
        """Allow instance to be called as a callback."""
        self.update(current)
