"""
Visualization helpers for vector indexing algorithm internals.

This module provides visualization functions for understanding and comparing
vector indexing algorithms, including tradeoff plots, algorithm structure
visualizations, and parameter tuning heatmaps.

Features:
- Recall vs latency tradeoff plots
- Memory vs accuracy Pareto frontiers
- Parameter sweep heatmaps
- Algorithm structure visualization (graphs, trees)
- Distance distribution plots

Author: Claude
Date: 2026-01-09
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from matplotlib.patches import FancyBboxPatch
import warnings

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def plot_recall_latency_tradeoff(
    results: List[Dict[str, Any]],
    recall_key: str = "recall@10",
    latency_key: str = "latency_p50",
    title: str = "Recall vs Latency Tradeoff",
    figsize: Tuple[int, int] = (10, 7),
    show_pareto: bool = True,
    annotate: bool = True
) -> plt.Figure:
    """
    Create scatter plot showing recall-latency tradeoff.

    Visualizes the classic tradeoff between search accuracy (recall) and
    speed (latency). Optionally highlights the Pareto frontier.

    Args:
        results: List of dictionaries with keys matching recall_key and latency_key
                 Each dict should also have 'name' key for labeling
        recall_key: Key name for recall values (e.g., "recall@10")
        latency_key: Key name for latency values (e.g., "latency_p50")
        title: Plot title
        figsize: Figure size as (width, height)
        show_pareto: If True, highlight Pareto-optimal points
        annotate: If True, add text labels for each point

    Returns:
        Matplotlib figure object

    Example:
        >>> results = [
        ...     {"name": "Flat", "recall@10": 1.0, "latency_p50": 80},
        ...     {"name": "HNSW", "recall@10": 0.97, "latency_p50": 10},
        ...     {"name": "IVF", "recall@10": 0.95, "latency_p50": 15}
        ... ]
        >>> fig = plot_recall_latency_tradeoff(results)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data
    names = [r['name'] for r in results]
    recalls = [r[recall_key] * 100 if r[recall_key] <= 1 else r[recall_key]
              for r in results]
    latencies = [r[latency_key] for r in results]

    # Create color map
    colors = sns.color_palette("husl", len(results))

    # Find Pareto frontier (maximize recall, minimize latency)
    if show_pareto:
        pareto_mask = _find_pareto_frontier(np.array(recalls), np.array(latencies))
        pareto_indices = np.where(pareto_mask)[0]
    else:
        pareto_indices = []

    # Plot points
    for i, (name, recall, latency) in enumerate(zip(names, recalls, latencies)):
        is_pareto = i in pareto_indices

        # Larger marker for Pareto-optimal points
        marker_size = 300 if is_pareto else 200
        edge_color = 'gold' if is_pareto else 'none'
        edge_width = 3 if is_pareto else 0

        ax.scatter(latency, recall,
                  s=marker_size, alpha=0.7,
                  c=[colors[i]], label=name,
                  edgecolors=edge_color, linewidths=edge_width,
                  zorder=3 if is_pareto else 2)

        # Annotate
        if annotate:
            ax.annotate(name,
                       xy=(latency, recall),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=10, alpha=0.9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))

    # Draw Pareto frontier line
    if show_pareto and len(pareto_indices) > 1:
        pareto_recalls = [recalls[i] for i in sorted(pareto_indices, key=lambda x: latencies[x])]
        pareto_latencies = [latencies[i] for i in sorted(pareto_indices, key=lambda x: latencies[x])]
        ax.plot(pareto_latencies, pareto_recalls,
               'k--', alpha=0.4, linewidth=2, label='Pareto Frontier', zorder=1)

    # Formatting
    ax.set_xlabel('Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Recall (%)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0, min(recalls) - 5), top=min(105, max(recalls) + 5))

    # Add "better" indicators
    ax.text(0.02, 0.98, '← Faster', transform=ax.transAxes,
           fontsize=11, va='top', ha='left', style='italic', alpha=0.6)
    ax.text(0.02, 0.02, '↑ More Accurate', transform=ax.transAxes,
           fontsize=11, va='bottom', ha='left', style='italic', alpha=0.6)

    plt.tight_layout()
    return fig


def _find_pareto_frontier(values_maximize: np.ndarray,
                         values_minimize: np.ndarray) -> np.ndarray:
    """
    Find Pareto-optimal points (maximize first, minimize second).

    Args:
        values_maximize: Values to maximize (e.g., recall)
        values_minimize: Values to minimize (e.g., latency)

    Returns:
        Boolean mask indicating Pareto-optimal points
    """
    n = len(values_maximize)
    pareto_mask = np.ones(n, dtype=bool)

    for i in range(n):
        if pareto_mask[i]:
            # Point i is dominated if there exists j with:
            # recall[j] >= recall[i] AND latency[j] <= latency[i]
            # (and at least one is strict inequality)
            dominated = (
                (values_maximize >= values_maximize[i]) &
                (values_minimize <= values_minimize[i]) &
                ((values_maximize > values_maximize[i]) | (values_minimize < values_minimize[i]))
            )
            pareto_mask[dominated] = False

    return pareto_mask


def plot_memory_vs_accuracy(
    results: List[Dict[str, Any]],
    memory_key: str = "memory_mb",
    accuracy_key: str = "recall@10",
    title: str = "Memory vs Accuracy Tradeoff",
    figsize: Tuple[int, int] = (10, 7)
) -> plt.Figure:
    """
    Plot memory footprint vs accuracy with Pareto frontier.

    Args:
        results: List of result dictionaries
        memory_key: Key for memory values
        accuracy_key: Key for accuracy values
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = [r['name'] for r in results]
    memory = [r[memory_key] for r in results]
    accuracy = [r[accuracy_key] * 100 if r[accuracy_key] <= 1 else r[accuracy_key]
               for r in results]

    colors = sns.color_palette("viridis", len(results))

    # Find Pareto frontier (maximize accuracy, minimize memory)
    pareto_mask = _find_pareto_frontier(np.array(accuracy), np.array(memory))
    pareto_indices = np.where(pareto_mask)[0]

    # Plot
    for i, (name, mem, acc) in enumerate(zip(names, memory, accuracy)):
        is_pareto = i in pareto_indices
        marker_size = 300 if is_pareto else 200
        edge_color = 'red' if is_pareto else 'none'
        edge_width = 3 if is_pareto else 0

        ax.scatter(mem, acc,
                  s=marker_size, alpha=0.7,
                  c=[colors[i]], label=name,
                  edgecolors=edge_color, linewidths=edge_width)

        ax.annotate(name, xy=(mem, acc),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, alpha=0.9)

    # Pareto line
    if len(pareto_indices) > 1:
        pareto_memory = [memory[i] for i in sorted(pareto_indices, key=lambda x: memory[x])]
        pareto_accuracy = [accuracy[i] for i in sorted(pareto_indices, key=lambda x: memory[x])]
        ax.plot(pareto_memory, pareto_accuracy,
               'r--', alpha=0.5, linewidth=2, label='Pareto Frontier')

    ax.set_xlabel('Memory (MB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0, min(accuracy) - 5), top=105)

    plt.tight_layout()
    return fig


def plot_parameter_sweep(
    param1_values: List,
    param2_values: List,
    metric_matrix: np.ndarray,
    param1_name: str,
    param2_name: str,
    metric_name: str,
    title: str = "Parameter Sweep",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "YlOrRd"
) -> plt.Figure:
    """
    Create heatmap for 2D parameter sweep results.

    Args:
        param1_values: Values for parameter 1 (x-axis)
        param2_values: Values for parameter 2 (y-axis)
        metric_matrix: 2D array of metric values, shape (len(param2), len(param1))
        param1_name: Name of parameter 1
        param2_name: Name of parameter 2
        metric_name: Name of the metric being plotted
        title: Plot title
        figsize: Figure size
        cmap: Colormap name

    Returns:
        Matplotlib figure object

    Example:
        >>> m_values = [8, 16, 32]
        >>> ef_values = [50, 100, 200]
        >>> recall_matrix = np.array([[0.85, 0.90, 0.93],
        ...                           [0.88, 0.93, 0.95],
        ...                           [0.90, 0.95, 0.97]])
        >>> fig = plot_parameter_sweep(m_values, ef_values, recall_matrix,
        ...                           "m", "ef_search", "Recall@10")
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(metric_matrix, cmap=cmap, aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(param1_values)))
    ax.set_yticks(np.arange(len(param2_values)))
    ax.set_xticklabels(param1_values)
    ax.set_yticklabels(param2_values)

    # Labels
    ax.set_xlabel(param1_name, fontsize=13, fontweight='bold')
    ax.set_ylabel(param2_name, fontsize=13, fontweight='bold')
    ax.set_title(f"{title}\n{metric_name}", fontsize=14, fontweight='bold', pad=20)

    # Add text annotations
    for i in range(len(param2_values)):
        for j in range(len(param1_values)):
            value = metric_matrix[i, j]
            # Choose text color based on background
            text_color = "white" if value > metric_matrix.max() * 0.6 else "black"
            text = ax.text(j, i, f"{value:.3f}",
                         ha="center", va="center",
                         color=text_color, fontsize=10, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name, rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    return fig


def plot_index_structure(
    adjacency_dict: Dict[int, List[int]],
    title: str = "Index Graph Structure",
    layout: str = "spring",
    figsize: Tuple[int, int] = (12, 10),
    max_nodes: int = 100,
    highlight_path: Optional[List[int]] = None
) -> plt.Figure:
    """
    Visualize graph-based index structure (HNSW, NSW).

    Args:
        adjacency_dict: Dictionary mapping node_id -> list of neighbor node_ids
        title: Plot title
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
        figsize: Figure size
        max_nodes: Maximum nodes to display (samples if exceeded)
        highlight_path: Optional list of node IDs to highlight as path

    Returns:
        Matplotlib figure object

    Example:
        >>> # Simple graph: 0 -> [1,2], 1 -> [2,3], 2 -> [3], 3 -> []
        >>> graph = {0: [1, 2], 1: [2, 3], 2: [3], 3: []}
        >>> fig = plot_index_structure(graph, highlight_path=[0, 1, 3])
    """
    try:
        import networkx as nx
    except ImportError:
        warnings.warn("NetworkX not installed. Cannot visualize graph structure.")
        return plt.figure()

    fig, ax = plt.subplots(figsize=figsize)

    # Sample nodes if too many
    if len(adjacency_dict) > max_nodes:
        sampled_nodes = np.random.choice(list(adjacency_dict.keys()),
                                        size=max_nodes, replace=False)
        adjacency_dict = {k: v for k, v in adjacency_dict.items() if k in sampled_nodes}

    # Create NetworkX graph
    G = nx.DiGraph()
    for node, neighbors in adjacency_dict.items():
        for neighbor in neighbors:
            if neighbor in adjacency_dict:  # Only add if neighbor is in sampled set
                G.add_edge(node, neighbor)

    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)

    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if highlight_path and node in highlight_path:
            node_colors.append('#ff6b6b')  # Red for path
        else:
            node_colors.append('#4ecdc4')  # Teal for regular

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                          node_size=300, alpha=0.8, ax=ax)

    # Draw edges
    edge_colors = []
    edge_widths = []
    for edge in G.edges():
        if highlight_path and edge[0] in highlight_path and edge[1] in highlight_path:
            # Check if this edge is part of the path
            try:
                idx = highlight_path.index(edge[0])
                if idx + 1 < len(highlight_path) and highlight_path[idx + 1] == edge[1]:
                    edge_colors.append('#ff6b6b')
                    edge_widths.append(3)
                else:
                    edge_colors.append('#95a5a6')
                    edge_widths.append(1)
            except ValueError:
                edge_colors.append('#95a5a6')
                edge_widths.append(1)
        else:
            edge_colors.append('#95a5a6')
            edge_widths.append(1)

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                          width=edge_widths, alpha=0.5,
                          arrows=True, arrowsize=10, ax=ax)

    # Draw labels for highlighted nodes
    if highlight_path:
        labels = {node: str(node) for node in highlight_path}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')

    # Add legend if path is highlighted
    if highlight_path:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6b6b',
                  markersize=10, label='Search Path'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ecdc4',
                  markersize=10, label='Other Nodes')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    return fig


def plot_distance_distribution(
    distances: np.ndarray,
    bins: int = 50,
    title: str = "Distance Distribution",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot distribution of distances between vectors.

    Useful for understanding dataset characteristics and index behavior.

    Args:
        distances: Array of distance values
        bins: Number of histogram bins
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram
    n, bins_edges, patches = ax.hist(distances, bins=bins, alpha=0.7,
                                     color='steelblue', edgecolor='black')

    # Add statistics
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    std_dist = np.std(distances)

    ax.axvline(mean_dist, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dist:.3f}')
    ax.axvline(median_dist, color='green', linestyle='--', linewidth=2, label=f'Median: {median_dist:.3f}')

    # Labels
    ax.set_xlabel('Distance', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title(f"{title}\nStd: {std_dist:.3f}", fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_build_time_scaling(
    dataset_sizes: List[int],
    build_times: Dict[str, List[float]],
    title: str = "Index Build Time Scaling",
    figsize: Tuple[int, int] = (10, 7)
) -> plt.Figure:
    """
    Plot how build time scales with dataset size for different indexes.

    Args:
        dataset_sizes: List of dataset sizes (x-axis)
        build_times: Dict mapping index_name -> list of build times
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure object

    Example:
        >>> sizes = [1000, 10000, 100000]
        >>> times = {
        ...     "Flat": [0.1, 0.5, 2.0],
        ...     "HNSW": [2.0, 15.0, 120.0],
        ...     "IVF": [1.0, 8.0, 60.0]
        ... }
        >>> fig = plot_build_time_scaling(sizes, times)
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("husl", len(build_times))

    for i, (index_name, times) in enumerate(build_times.items()):
        ax.plot(dataset_sizes, times, marker='o', linewidth=2,
               markersize=8, label=index_name, color=colors[i])

    ax.set_xlabel('Dataset Size', fontsize=13, fontweight='bold')
    ax.set_ylabel('Build Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    print("Testing visualization helpers...")

    # Test recall-latency tradeoff
    results = [
        {"name": "Flat", "recall@10": 1.0, "latency_p50": 80, "memory_mb": 300},
        {"name": "HNSW", "recall@10": 0.97, "latency_p50": 10, "memory_mb": 500},
        {"name": "IVF-Flat", "recall@10": 0.98, "latency_p50": 15, "memory_mb": 310},
        {"name": "IVF-PQ", "recall@10": 0.95, "latency_p50": 8, "memory_mb": 50},
        {"name": "ANNOY", "recall@10": 0.93, "latency_p50": 12, "memory_mb": 350}
    ]

    fig1 = plot_recall_latency_tradeoff(results)
    plt.savefig('/tmp/recall_latency.png', dpi=150, bbox_inches='tight')
    print("✅ Created recall-latency tradeoff plot")

    fig2 = plot_memory_vs_accuracy(results)
    plt.savefig('/tmp/memory_accuracy.png', dpi=150, bbox_inches='tight')
    print("✅ Created memory-accuracy tradeoff plot")

    # Test parameter sweep
    m_values = [8, 16, 32]
    ef_values = [50, 100, 200]
    recall_matrix = np.array([[0.85, 0.90, 0.93],
                             [0.88, 0.93, 0.95],
                             [0.90, 0.95, 0.97]])

    fig3 = plot_parameter_sweep(m_values, ef_values, recall_matrix,
                                "m", "ef_search", "Recall@10")
    plt.savefig('/tmp/param_sweep.png', dpi=150, bbox_inches='tight')
    print("✅ Created parameter sweep heatmap")

    print("\nAll visualizations created successfully!")
