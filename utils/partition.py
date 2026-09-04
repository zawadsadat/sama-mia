"""
Non-IID data partitioning for federated learning experiments.

Implements Dirichlet label-skew partitioning (Hsu et al., 2019):
    For each class c, sample a proportion vector p_c ~ Dir(α)
    and distribute class-c samples across K clients according to p_c.

Low α  → extreme heterogeneity (clients have mostly one class)
High α → nearly IID (clients have similar class distributions)
"""

import numpy as np


def dirichlet_partition(y, num_clients, alpha, seed=42):
    """
    Partition dataset indices using Dirichlet label skew.

    Args:
        y:           numpy array of labels (0 or 1)
        num_clients: number of FL clients (K)
        alpha:       Dirichlet concentration parameter
        seed:        random seed

    Returns:
        client_indices: list of K numpy arrays, each containing
                        sample indices for that client
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    n_classes = len(classes)

    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        class_idx = np.where(y == c)[0]
        rng.shuffle(class_idx)

        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Ensure minimum 1 sample per client per class (if possible)
        proportions = np.maximum(proportions, 1e-10)
        proportions = proportions / proportions.sum()

        # Split indices according to proportions
        splits = (proportions * len(class_idx)).astype(int)

        # Distribute any remainder
        remainder = len(class_idx) - splits.sum()
        for i in range(remainder):
            splits[i % num_clients] += 1

        current = 0
        for k in range(num_clients):
            client_indices[k].extend(class_idx[current:current + splits[k]])
            current += splits[k]

    # Convert to numpy arrays and shuffle within each client
    for k in range(num_clients):
        if len(client_indices[k]) > 0:
            client_indices[k] = np.array(client_indices[k], dtype=np.int64)
            rng.shuffle(client_indices[k])
        else:
            client_indices[k] = np.array([], dtype=np.int64)

    return client_indices


def get_partition_stats(y, client_indices):
    """
    Compute statistics about a partition for logging.

    Returns dict with per-client counts and class distributions.
    """
    stats = {
        "num_clients": len(client_indices),
        "clients": [],
    }

    for k, idx in enumerate(client_indices):
        labels = y[idx]
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        total = len(labels)
        pos_rate = n_pos / total if total > 0 else 0

        stats["clients"].append({
            "id": k,
            "total": total,
            "positive": n_pos,
            "negative": n_neg,
            "pos_rate": pos_rate,
        })

    # Compute heterogeneity metric: std of positive rates across clients
    pos_rates = [c["pos_rate"] for c in stats["clients"]]
    stats["pos_rate_mean"] = np.mean(pos_rates)
    stats["pos_rate_std"] = np.std(pos_rates)
    stats["min_samples"] = min(c["total"] for c in stats["clients"])
    stats["max_samples"] = max(c["total"] for c in stats["clients"])

    return stats


def print_partition_stats(stats, alpha):
    """Pretty print partition statistics."""
    K = stats["num_clients"]
    print(f"\n  Partition: α={alpha}, K={K}")
    print(f"  {'Client':<10} {'Total':>8} {'Pos':>6} {'Neg':>6} {'Pos%':>8}")
    print(f"  {'-'*40}")
    for c in stats["clients"]:
        print(f"  Client {c['id']:<3} {c['total']:>8} {c['positive']:>6} "
              f"{c['negative']:>6} {c['pos_rate']:>7.1%}")
    print(f"  {'-'*40}")
    print(f"  Pos rate: mean={stats['pos_rate_mean']:.3f}, std={stats['pos_rate_std']:.3f}")
    print(f"  Samples: min={stats['min_samples']}, max={stats['max_samples']}")


if __name__ == "__main__":
    # Quick demo
    np.random.seed(0)
    y_demo = np.concatenate([np.zeros(900), np.ones(100)])  # 10% positive

    for alpha in [0.1, 0.5, 1.0, 10.0]:
        idx = dirichlet_partition(y_demo, num_clients=5, alpha=alpha)
        stats = get_partition_stats(y_demo, idx)
        print_partition_stats(stats, alpha)
