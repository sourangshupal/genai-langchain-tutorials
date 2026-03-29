"""
Shared benchmarking framework for consistent metrics across all indexing notebooks.

This module provides a unified interface for benchmarking different vector indexing
strategies, ensuring consistent measurement and comparison across notebooks.

Key Features:
- Build time measurement
- Search latency profiling (p50, p95, p99)
- Recall@k calculation against ground truth
- Memory usage tracking
- Comprehensive reporting and visualization

Author: Claude
Date: 2026-01-09
"""

import time
import numpy as np
import psutil
import os
from typing import Callable, Dict, List, Tuple, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field


@dataclass
class BenchmarkResults:
    """Container for benchmark metrics"""
    index_name: str
    build_time_seconds: float = 0.0
    search_latency_ms: Dict[str, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    memory_usage_mb: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for easy serialization"""
        return {
            'index_name': self.index_name,
            'build_time_seconds': self.build_time_seconds,
            'search_latency_ms': self.search_latency_ms,
            'recall_at_k': self.recall_at_k,
            'memory_usage_mb': self.memory_usage_mb,
            'metadata': self.metadata
        }


class IndexBenchmark:
    """
    Benchmark framework for vector indexing strategies.

    This class provides methods to measure and compare different aspects of
    vector index performance including build time, search latency, recall
    accuracy, and memory usage.

    Args:
        index_name: Descriptive name for the index being benchmarked
        ground_truth_results: Optional ground truth nearest neighbor results
                             for recall calculation. Should be a 2D array of
                             shape (num_queries, k) containing indices.

    Example:
        >>> benchmark = IndexBenchmark("HNSW-M16", ground_truth)
        >>> build_time = benchmark.measure_build_time(lambda: build_index())
        >>> latency = benchmark.measure_search_latency(search_fn, queries)
        >>> recall = benchmark.calculate_recall_at_k(results, k_values=[1,5,10])
        >>> report = benchmark.generate_report()
    """

    def __init__(self,
                 index_name: str,
                 ground_truth_results: Optional[np.ndarray] = None):
        self.index_name = index_name
        self.ground_truth = ground_truth_results
        self.results = BenchmarkResults(index_name=index_name)

    def measure_build_time(self, build_fn: Callable, *args, **kwargs) -> float:
        """
        Measure index construction time.

        Args:
            build_fn: Function that builds the index
            *args, **kwargs: Arguments to pass to build_fn

        Returns:
            Build time in seconds

        Example:
            >>> def build_index():
            ...     index = faiss.IndexHNSWFlat(768, 16)
            ...     index.add(vectors)
            ...     return index
            >>> build_time = benchmark.measure_build_time(build_index)
        """
        start_time = time.perf_counter()
        result = build_fn(*args, **kwargs)
        end_time = time.perf_counter()

        build_time = end_time - start_time
        self.results.build_time_seconds = build_time

        print(f"✅ Index built in {build_time:.2f} seconds")
        return build_time

    def measure_search_latency(self,
                               search_fn: Callable,
                               queries: np.ndarray,
                               k: int = 10,
                               warmup_queries: int = 10) -> Dict[str, float]:
        """
        Measure search latency statistics.

        Performs warmup queries to stabilize caching, then measures latency
        for all queries and computes percentile statistics.

        Args:
            search_fn: Function that takes a query vector and returns results
                      Should accept (query_vector, k) as arguments
            queries: Array of query vectors, shape (num_queries, dim)
            k: Number of nearest neighbors to retrieve
            warmup_queries: Number of queries to run for warmup (not measured)

        Returns:
            Dictionary with keys: 'p50', 'p95', 'p99', 'mean', 'std', 'min', 'max'
            All values are in milliseconds

        Example:
            >>> def search(q, k):
            ...     distances, indices = index.search(q.reshape(1, -1), k)
            ...     return indices[0]
            >>> latency = benchmark.measure_search_latency(search, query_vectors)
        """
        num_queries = len(queries)

        # Warmup phase
        warmup_count = min(warmup_queries, num_queries)
        for i in range(warmup_count):
            _ = search_fn(queries[i], k)

        # Measurement phase
        latencies = []
        for query in queries:
            start = time.perf_counter()
            _ = search_fn(query, k)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to milliseconds

        latencies = np.array(latencies)

        stats = {
            'p50': float(np.percentile(latencies, 50)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
            'mean': float(np.mean(latencies)),
            'std': float(np.std(latencies)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies))
        }

        self.results.search_latency_ms = stats

        print(f"✅ Search latency: p50={stats['p50']:.2f}ms, "
              f"p95={stats['p95']:.2f}ms, p99={stats['p99']:.2f}ms")

        return stats

    def calculate_recall_at_k(self,
                              retrieved_results: np.ndarray,
                              k_values: List[int] = [1, 5, 10, 50]) -> Dict[int, float]:
        """
        Calculate recall@k against ground truth.

        Recall@k measures what fraction of the true k nearest neighbors
        are found in the top k retrieved results.

        Args:
            retrieved_results: 2D array of retrieved indices,
                              shape (num_queries, max_k)
            k_values: List of k values to compute recall for

        Returns:
            Dictionary mapping k -> recall@k (as fraction 0.0-1.0)

        Raises:
            ValueError: If ground truth was not provided during initialization

        Example:
            >>> # retrieved_results[i] contains indices of top-k results for query i
            >>> recall = benchmark.calculate_recall_at_k(retrieved_results, [1,5,10])
            >>> print(f"Recall@10: {recall[10]:.2%}")
        """
        if self.ground_truth is None:
            raise ValueError("Ground truth results not provided. "
                           "Initialize IndexBenchmark with ground_truth_results.")

        num_queries = len(retrieved_results)
        recalls = {}

        for k in k_values:
            if k > retrieved_results.shape[1]:
                print(f"⚠️  Warning: k={k} exceeds retrieved results size, skipping")
                continue

            total_recall = 0.0
            for i in range(num_queries):
                # Get top-k from both retrieved and ground truth
                retrieved_set = set(retrieved_results[i, :k])
                ground_truth_set = set(self.ground_truth[i, :k])

                # Calculate overlap
                intersection = len(retrieved_set & ground_truth_set)
                recall = intersection / k
                total_recall += recall

            avg_recall = total_recall / num_queries
            recalls[k] = avg_recall

        self.results.recall_at_k = recalls

        print(f"✅ Recall@k computed: " +
              ", ".join([f"R@{k}={v:.2%}" for k, v in recalls.items()]))

        return recalls

    def measure_memory_usage(self,
                            index_object: Any = None,
                            index_file_path: Optional[str] = None) -> Dict[str, float]:
        """
        Measure memory usage of the index.

        Args:
            index_object: The index object in memory (for RAM measurement)
            index_file_path: Path to serialized index file (for disk measurement)

        Returns:
            Dictionary with keys: 'ram_mb' (current process), 'index_size_mb' (disk)

        Example:
            >>> memory = benchmark.measure_memory_usage(
            ...     index_object=index,
            ...     index_file_path="index.faiss"
            ... )
        """
        memory_stats = {}

        # Measure current process RAM usage
        process = psutil.Process(os.getpid())
        ram_bytes = process.memory_info().rss
        memory_stats['ram_mb'] = ram_bytes / (1024 * 1024)

        # Measure index file size if provided
        if index_file_path and os.path.exists(index_file_path):
            file_size_bytes = os.path.getsize(index_file_path)
            memory_stats['index_size_mb'] = file_size_bytes / (1024 * 1024)

        # Attempt to estimate index object size (approximation)
        if index_object is not None:
            try:
                import sys
                obj_size_bytes = sys.getsizeof(index_object)
                memory_stats['object_size_mb'] = obj_size_bytes / (1024 * 1024)
            except:
                pass  # Some objects don't support getsizeof

        self.results.memory_usage_mb = memory_stats

        print(f"✅ Memory usage: " +
              ", ".join([f"{k}={v:.2f}MB" for k, v in memory_stats.items()]))

        return memory_stats

    def generate_report(self) -> Dict:
        """
        Generate comprehensive benchmark report.

        Returns:
            Complete benchmark results as dictionary

        Example:
            >>> report = benchmark.generate_report()
            >>> print(f"Index: {report['index_name']}")
            >>> print(f"Build time: {report['build_time_seconds']:.2f}s")
        """
        report = self.results.to_dict()

        print(f"\n{'='*60}")
        print(f"📊 Benchmark Report: {self.index_name}")
        print(f"{'='*60}")

        if self.results.build_time_seconds > 0:
            print(f"\n⏱️  Build Time: {self.results.build_time_seconds:.2f}s")

        if self.results.search_latency_ms:
            print(f"\n🔍 Search Latency:")
            for metric, value in self.results.search_latency_ms.items():
                print(f"  {metric:>6s}: {value:>8.2f}ms")

        if self.results.recall_at_k:
            print(f"\n🎯 Recall@k:")
            for k, recall in sorted(self.results.recall_at_k.items()):
                print(f"  R@{k:>2d}: {recall:>6.2%}")

        if self.results.memory_usage_mb:
            print(f"\n💾 Memory Usage:")
            for metric, value in self.results.memory_usage_mb.items():
                print(f"  {metric:>16s}: {value:>8.2f}MB")

        print(f"{'='*60}\n")

        return report


def compare_indexes(benchmark_results: List[BenchmarkResults],
                   figsize: Tuple[int, int] = (18, 5)) -> plt.Figure:
    """
    Create comprehensive comparison visualizations for multiple indexes.

    Generates a 3-panel figure comparing:
    1. Recall vs Latency tradeoff
    2. Memory usage comparison
    3. Build time comparison

    Args:
        benchmark_results: List of BenchmarkResults objects
        figsize: Figure size as (width, height)

    Returns:
        Matplotlib figure object

    Example:
        >>> results = [benchmark1.results, benchmark2.results, benchmark3.results]
        >>> fig = compare_indexes(results)
        >>> plt.show()
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Extract data for plotting
    names = [r.index_name for r in benchmark_results]
    colors = sns.color_palette("husl", len(benchmark_results))

    # Panel 1: Recall vs Latency
    ax1 = axes[0]
    for i, result in enumerate(benchmark_results):
        if result.recall_at_k and result.search_latency_ms:
            # Use recall@10 as representative metric
            recall_10 = result.recall_at_k.get(10, 0)
            latency_p50 = result.search_latency_ms.get('p50', 0)

            ax1.scatter(latency_p50, recall_10 * 100,
                       s=200, alpha=0.6, c=[colors[i]],
                       label=result.index_name)
            ax1.annotate(result.index_name,
                        xy=(latency_p50, recall_10 * 100),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)

    ax1.set_xlabel('Latency (ms, p50)', fontsize=11)
    ax1.set_ylabel('Recall@10 (%)', fontsize=11)
    ax1.set_title('Recall vs Latency Tradeoff', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0, top=105)

    # Panel 2: Memory Usage
    ax2 = axes[1]
    memory_values = []
    for result in benchmark_results:
        if result.memory_usage_mb:
            # Prefer index_size_mb, fallback to ram_mb
            memory = result.memory_usage_mb.get('index_size_mb',
                     result.memory_usage_mb.get('ram_mb', 0))
            memory_values.append(memory)
        else:
            memory_values.append(0)

    bars = ax2.bar(range(len(names)), memory_values, color=colors, alpha=0.6)
    ax2.set_xlabel('Index', fontsize=11)
    ax2.set_ylabel('Memory Usage (MB)', fontsize=11)
    ax2.set_title('Memory Footprint Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, memory_values):
        if value > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.0f}MB', ha='center', va='bottom', fontsize=9)

    # Panel 3: Build Time
    ax3 = axes[2]
    build_times = [r.build_time_seconds for r in benchmark_results]
    bars = ax3.bar(range(len(names)), build_times, color=colors, alpha=0.6)
    ax3.set_xlabel('Index', fontsize=11)
    ax3.set_ylabel('Build Time (seconds)', fontsize=11)
    ax3.set_title('Index Construction Time', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax3.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, build_times):
        if value > 0:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}s', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def plot_recall_curves(benchmark_results: List[BenchmarkResults],
                      k_values: Optional[List[int]] = None,
                      figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot recall@k curves for multiple indexes.

    Args:
        benchmark_results: List of BenchmarkResults objects
        k_values: List of k values to plot (if None, uses all available)
        figsize: Figure size as (width, height)

    Returns:
        Matplotlib figure object

    Example:
        >>> results = [benchmark1.results, benchmark2.results]
        >>> fig = plot_recall_curves(results, k_values=[1, 5, 10, 20, 50])
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("husl", len(benchmark_results))

    for i, result in enumerate(benchmark_results):
        if not result.recall_at_k:
            continue

        # Sort by k for proper line plotting
        k_vals = sorted(result.recall_at_k.keys())
        recalls = [result.recall_at_k[k] * 100 for k in k_vals]

        # Filter to specified k_values if provided
        if k_values:
            filtered_k = [k for k in k_vals if k in k_values]
            filtered_recalls = [result.recall_at_k[k] * 100 for k in filtered_k]
            k_vals, recalls = filtered_k, filtered_recalls

        ax.plot(k_vals, recalls, marker='o', linewidth=2,
               markersize=8, label=result.index_name,
               color=colors[i], alpha=0.7)

    ax.set_xlabel('k (number of neighbors)', fontsize=12)
    ax.set_ylabel('Recall@k (%)', fontsize=12)
    ax.set_title('Recall@k Comparison Across Indexes', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    return fig


def create_summary_table(benchmark_results: List[BenchmarkResults]) -> str:
    """
    Create a formatted summary table of benchmark results.

    Args:
        benchmark_results: List of BenchmarkResults objects

    Returns:
        Formatted string table suitable for printing or markdown

    Example:
        >>> table = create_summary_table([b1.results, b2.results, b3.results])
        >>> print(table)
    """
    # Header
    table = "\n" + "="*100 + "\n"
    table += f"{'Index':<20} | {'Build(s)':>10} | {'Latency p50(ms)':>15} | "
    table += f"{'Recall@10':>12} | {'Memory(MB)':>12}\n"
    table += "="*100 + "\n"

    # Rows
    for result in benchmark_results:
        name = result.index_name[:20]
        build_time = f"{result.build_time_seconds:.2f}" if result.build_time_seconds else "N/A"
        latency = f"{result.search_latency_ms.get('p50', 0):.2f}" if result.search_latency_ms else "N/A"
        recall = f"{result.recall_at_k.get(10, 0):.2%}" if result.recall_at_k else "N/A"
        memory = result.memory_usage_mb.get('index_size_mb', result.memory_usage_mb.get('ram_mb', 0))
        memory_str = f"{memory:.0f}" if memory else "N/A"

        table += f"{name:<20} | {build_time:>10} | {latency:>15} | {recall:>12} | {memory_str:>12}\n"

    table += "="*100 + "\n"

    return table
