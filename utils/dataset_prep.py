"""
Dataset preparation with caching for vector indexing notebooks.

This module handles loading the climate_fever dataset, generating embeddings
using BGE-base-en-v1.5, and computing ground truth nearest neighbors for
benchmark evaluation.

Features:
- Automatic dataset download from Hugging Face
- Embedding generation with caching to avoid recomputation
- Ground truth computation using exact brute-force search
- Test query generation
- Disk-based caching for fast reloading

Author: Claude
Date: 2026-01-09
"""

import numpy as np
import os
import pickle
from typing import Tuple, List, Optional, Dict
from pathlib import Path
from tqdm.auto import tqdm


def prepare_climate_fever_vectors(
    sample_size: int = 100000,
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    cache_dir: str = "./data/embeddings",
    force_recompute: bool = False,
    num_queries: int = 100,
    ground_truth_k: int = 50,
    random_seed: int = 42
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Prepare climate_fever dataset with BGE embeddings and ground truth.

    This function loads the climate_fever dataset, generates embeddings using
    the specified model, and computes exact nearest neighbors as ground truth
    for recall evaluation. Results are cached to disk for fast reloading.

    Args:
        sample_size: Number of documents to use (max ~1.3M available)
        embedding_model: HuggingFace model identifier for embeddings
        cache_dir: Directory to store cached embeddings
        force_recompute: If True, ignore cache and recompute everything
        num_queries: Number of test queries to generate
        ground_truth_k: Number of nearest neighbors to compute for ground truth
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of:
        - vectors: Document embeddings, shape (sample_size, embedding_dim)
        - texts: Original text strings for documents
        - query_vectors: Test query embeddings, shape (num_queries, embedding_dim)
        - ground_truth_indices: Exact nearest neighbor indices,
                               shape (num_queries, ground_truth_k)

    Example:
        >>> vectors, texts, queries, ground_truth = prepare_climate_fever_vectors(
        ...     sample_size=100000,
        ...     num_queries=100
        ... )
        >>> print(f"Dataset: {vectors.shape[0]} docs, {vectors.shape[1]} dims")
        >>> print(f"Queries: {queries.shape[0]} test queries")
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Generate cache file name based on parameters
    cache_file = cache_path / f"climate_fever_{sample_size}_{embedding_model.replace('/', '_')}.pkl"

    # Check if cache exists
    if cache_file.exists() and not force_recompute:
        print(f"📂 Loading cached embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        print(f"✅ Loaded {len(cached_data['vectors'])} document vectors")
        print(f"✅ Loaded {len(cached_data['query_vectors'])} query vectors")
        print(f"✅ Ground truth shape: {cached_data['ground_truth_indices'].shape}")

        return (
            cached_data['vectors'],
            cached_data['texts'],
            cached_data['query_vectors'],
            cached_data['ground_truth_indices']
        )

    # If not cached, generate embeddings
    print(f"🔄 Cache not found or force_recompute=True, generating embeddings...")
    print(f"   Model: {embedding_model}")
    print(f"   Sample size: {sample_size:,}")

    # Load dataset
    print(f"\n📥 Loading climate_fever dataset...")
    vectors, texts = _load_climate_fever_dataset(sample_size)

    # Generate embeddings
    print(f"\n🔮 Generating embeddings using {embedding_model}...")
    vectors = _generate_embeddings(texts, embedding_model)

    # Generate test queries (sample from dataset)
    print(f"\n🔍 Generating {num_queries} test queries...")
    query_indices = np.random.choice(len(vectors), size=num_queries, replace=False)
    query_vectors = vectors[query_indices].copy()
    query_texts = [texts[i] for i in query_indices]

    # Compute ground truth using brute force search
    print(f"\n🎯 Computing ground truth (exact nearest neighbors, k={ground_truth_k})...")
    ground_truth_indices = _compute_ground_truth(vectors, query_vectors, ground_truth_k)

    # Cache the results
    print(f"\n💾 Caching results to {cache_file}...")
    cached_data = {
        'vectors': vectors,
        'texts': texts,
        'query_vectors': query_vectors,
        'query_texts': query_texts,
        'query_indices': query_indices,
        'ground_truth_indices': ground_truth_indices,
        'metadata': {
            'sample_size': sample_size,
            'embedding_model': embedding_model,
            'num_queries': num_queries,
            'ground_truth_k': ground_truth_k,
            'embedding_dim': vectors.shape[1],
            'random_seed': random_seed
        }
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Dataset preparation complete!")
    print(f"   Documents: {vectors.shape}")
    print(f"   Queries: {query_vectors.shape}")
    print(f"   Ground truth: {ground_truth_indices.shape}")

    return vectors, texts, query_vectors, ground_truth_indices


def _load_climate_fever_dataset(sample_size: int) -> Tuple[np.ndarray, List[str]]:
    """
    Load climate_fever dataset from HuggingFace.

    Args:
        sample_size: Number of samples to load

    Returns:
        Tuple of (vectors_placeholder, texts)
        Note: vectors_placeholder is empty, actual embeddings generated later
    """
    try:
        from datasets import load_dataset

        print("   Loading from HuggingFace datasets...")
        dataset = load_dataset("tals/climate_fever", split="test")

        # Sample if needed
        if sample_size < len(dataset):
            indices = np.random.choice(len(dataset), size=sample_size, replace=False)
            dataset = dataset.select(indices)

        # Extract texts (claim + evidence)
        texts = []
        for item in tqdm(dataset, desc="Processing texts"):
            # Combine claim and evidence for richer context
            claim = item.get('claim', '')
            evidence = item.get('evidence', '')
            combined_text = f"{claim} {evidence}".strip()
            if combined_text:
                texts.append(combined_text)

        print(f"   ✅ Loaded {len(texts)} text documents")

        # Placeholder for vectors (will be populated by embedding function)
        vectors = np.array([])

        return vectors, texts

    except ImportError:
        print("⚠️  'datasets' library not found. Using synthetic data...")
        return _generate_synthetic_dataset(sample_size)


def _generate_synthetic_dataset(sample_size: int, text_length: int = 50) -> Tuple[np.ndarray, List[str]]:
    """
    Generate synthetic dataset for testing when real data is unavailable.

    Args:
        sample_size: Number of documents to generate
        text_length: Approximate number of words per document

    Returns:
        Tuple of (empty vectors array, synthetic texts)
    """
    import random
    import string

    print(f"   Generating {sample_size} synthetic documents...")

    # Common climate-related words for more realistic synthetic data
    climate_words = [
        "climate", "temperature", "warming", "carbon", "emissions", "greenhouse",
        "atmosphere", "ocean", "ice", "arctic", "antarctic", "fossil", "renewable",
        "solar", "wind", "energy", "pollution", "deforestation", "biodiversity"
    ]

    texts = []
    for i in range(sample_size):
        # Generate random text with climate-related words
        words = [random.choice(climate_words) if random.random() < 0.3
                else ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
                for _ in range(text_length)]
        text = ' '.join(words)
        texts.append(text)

    print(f"   ✅ Generated {len(texts)} synthetic documents")

    return np.array([]), texts


def _generate_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    """
    Generate embeddings for texts using the specified model.

    Args:
        texts: List of text strings
        model_name: HuggingFace model identifier

    Returns:
        Embeddings array of shape (len(texts), embedding_dim)
    """
    try:
        from sentence_transformers import SentenceTransformer

        print(f"   Loading model: {model_name}...")
        model = SentenceTransformer(model_name)

        print(f"   Encoding {len(texts)} documents...")
        # Use show_progress_bar for visual feedback
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        print(f"   ✅ Generated embeddings: {embeddings.shape}")

        return embeddings

    except ImportError:
        print("⚠️  'sentence-transformers' not found. Using random embeddings...")
        return _generate_random_embeddings(len(texts), embedding_dim=768)


def _generate_random_embeddings(num_vectors: int, embedding_dim: int = 768) -> np.ndarray:
    """
    Generate random normalized embeddings for testing.

    Args:
        num_vectors: Number of vectors to generate
        embedding_dim: Dimension of each vector

    Returns:
        Random normalized embeddings
    """
    print(f"   Generating {num_vectors} random vectors ({embedding_dim}D)...")

    embeddings = np.random.randn(num_vectors, embedding_dim).astype(np.float32)

    # Normalize to unit length (for cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    print(f"   ✅ Generated random embeddings: {embeddings.shape}")

    return embeddings


def _compute_ground_truth(vectors: np.ndarray,
                         query_vectors: np.ndarray,
                         k: int) -> np.ndarray:
    """
    Compute exact nearest neighbors using brute force search.

    This provides the ground truth for recall@k evaluation.

    Args:
        vectors: Document vectors, shape (N, D)
        query_vectors: Query vectors, shape (Q, D)
        k: Number of nearest neighbors to find

    Returns:
        Indices of k nearest neighbors for each query, shape (Q, k)
    """
    num_queries = len(query_vectors)
    ground_truth = np.zeros((num_queries, k), dtype=np.int64)

    print(f"   Computing for {num_queries} queries...")

    for i in tqdm(range(num_queries), desc="Computing ground truth"):
        query = query_vectors[i]

        # Compute cosine similarity (since vectors are normalized)
        # similarity = query · vectors^T
        similarities = np.dot(vectors, query)

        # Get top-k indices (argsort returns ascending, so we take last k and reverse)
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        ground_truth[i] = top_k_indices

    print(f"   ✅ Ground truth computed: {ground_truth.shape}")

    return ground_truth


def load_cached_dataset(
    sample_size: int = 100000,
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    cache_dir: str = "./data/embeddings"
) -> Optional[Dict]:
    """
    Load cached dataset if available.

    Args:
        sample_size: Sample size used during caching
        embedding_model: Model name used for embeddings
        cache_dir: Cache directory

    Returns:
        Cached data dictionary or None if not found
    """
    cache_path = Path(cache_dir)
    cache_file = cache_path / f"climate_fever_{sample_size}_{embedding_model.replace('/', '_')}.pkl"

    if cache_file.exists():
        print(f"📂 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"⚠️  No cached data found at {cache_file}")
    return None


def clear_cache(cache_dir: str = "./data/embeddings"):
    """
    Clear all cached embeddings.

    Args:
        cache_dir: Cache directory to clear
    """
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        print(f"⚠️  Cache directory does not exist: {cache_dir}")
        return

    cache_files = list(cache_path.glob("*.pkl"))

    if not cache_files:
        print(f"ℹ️  No cache files found in {cache_dir}")
        return

    print(f"🗑️  Clearing {len(cache_files)} cache file(s)...")

    for cache_file in cache_files:
        cache_file.unlink()
        print(f"   Deleted: {cache_file.name}")

    print(f"✅ Cache cleared")


def get_dataset_info(
    sample_size: int = 100000,
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    cache_dir: str = "./data/embeddings"
) -> Dict:
    """
    Get information about cached dataset.

    Args:
        sample_size: Sample size to query
        embedding_model: Model name to query
        cache_dir: Cache directory

    Returns:
        Dictionary with dataset metadata
    """
    cached_data = load_cached_dataset(sample_size, embedding_model, cache_dir)

    if cached_data is None:
        return {"status": "not_cached", "message": "Dataset not found in cache"}

    metadata = cached_data.get('metadata', {})
    vectors = cached_data.get('vectors', np.array([]))
    queries = cached_data.get('query_vectors', np.array([]))
    ground_truth = cached_data.get('ground_truth_indices', np.array([]))

    info = {
        "status": "cached",
        "num_documents": len(vectors),
        "num_queries": len(queries),
        "embedding_dim": vectors.shape[1] if len(vectors) > 0 else 0,
        "ground_truth_k": ground_truth.shape[1] if len(ground_truth) > 0 else 0,
        "metadata": metadata
    }

    return info


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Climate Fever Dataset Preparation")
    print("="*60)

    # Prepare dataset (will use cache if available)
    vectors, texts, query_vectors, ground_truth = prepare_climate_fever_vectors(
        sample_size=1000,  # Small sample for testing
        num_queries=10,
        ground_truth_k=10
    )

    print(f"\n📊 Dataset Summary:")
    print(f"   Documents: {len(vectors):,} × {vectors.shape[1]}D")
    print(f"   Queries: {len(query_vectors):,}")
    print(f"   Ground truth: {ground_truth.shape}")
    print(f"   Sample text: {texts[0][:100]}...")
