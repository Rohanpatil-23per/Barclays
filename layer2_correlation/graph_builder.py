import pandas as pd
import numpy as np
import torch
import pickle
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# WHY WE BUILD GRAPHS:
# A flat CSV row tells us about ONE alert in isolation.
# GATv2 needs RELATIONSHIPS between alerts — how one event
# leads to another. Graphs capture this structure.
# Node = one alert. Edge = relationship between alerts.
#
# KEY FIX IN THIS VERSION — MIXED GRAPHS:
# Each graph contains a random mix of attack AND benign nodes.
# Graph label = 1.0 if more than 20% of nodes are attacks.
#
# Why mixed:
# 1. Realistic — real SOC alerts arrive mixed, not sorted
# 2. Prevents feature variance shortcut — attack nodes had
#    std=0.000 for rst_flag_count, giving model a cheat signal
# 3. Forces GATv2 to identify anomalous nodes within normal
#    traffic — the actual security task in banking
# 4. Produces calibrated confidence scores (not always 1.0)
#    which downstream DQN and fidelity ranking need
# ─────────────────────────────────────────────────────────────


def build_knn_edges(n, k, max_nodes):
    """
    Builds k-nearest neighbor edges for a graph of n nodes.

    Every graph — attack AND benign — gets identical edges.
    The only edge feature is normalized position distance.
    No confidence value, no class signal in edges.

    Why: any difference in edge structure between classes
    gives the model a trivial shortcut. We want it to learn
    from actual network flow values at each node.

    n         : number of nodes in the graph
    k         : number of neighbors per node
    max_nodes : used to normalize distance to 0-1 range
    """
    edge_index = []
    edge_attr  = []

    k = min(k, n - 1)

    for i in range(n):
        for j in range(max(0, i - k), min(n, i + k + 1)):
            if i == j:
                continue
            edge_index.append([i, j])

            # Single edge feature: normalized position distance
            # Identical structure for ALL graphs — no shortcuts
            dist = abs(i - j) / max_nodes
            edge_attr.append([dist])

    return edge_index, edge_attr


def drop_edge(g, p=0.15):
    """
    Randomly removes p% of edges from a graph copy.
    Used for augmentation — model sees varied versions
    of the same pattern, preventing overfitting.
    Original graph is never modified (we use g.clone()).
    """
    g2   = g.clone()
    mask = torch.rand(g.edge_index.shape[1]) > p
    g2.edge_index = g.edge_index[:, mask]
    if g.edge_attr is not None:
        g2.edge_attr = g.edge_attr[mask]
    return g2


def build_mixed_graphs(df, feature_cols,
                        max_nodes_per_graph=50,
                        min_nodes_per_graph=5,
                        n_graphs=50000):
    """
    Builds graphs where each graph contains a MIX of
    attack and benign alert rows.

    For each graph:
    1. Randomly pick attack_ratio between 0% and 80%
    2. Sample that many attack rows + rest as benign
    3. Shuffle rows so attack nodes aren't clustered
    4. Label graph as attack if >20% nodes are attacks

    This approach:
    - Prevents feature variance shortcuts
    - Reflects real SOC alert streams (mixed traffic)
    - Forces model to use GATv2 attention to find anomalies
    - Produces meaningful 0-1 confidence scores
    """
    print(f"Building {n_graphs} mixed graphs...")

    # Separate attack and benign rows for sampling
    attack_df = df[df['is_attack'] == 1].reset_index(drop=True)
    benign_df = df[df['is_attack'] == 0].reset_index(drop=True)

    print(f"  Attack rows available: {len(attack_df):,}")
    print(f"  Benign rows available: {len(benign_df):,}")

    graphs  = []
    np.random.seed(42)

    for i in range(n_graphs):

        # Random attack ratio: 0% to 80% of nodes are attacks
        # Uniform distribution ensures model sees all ratios
        # including borderline cases near the 20% threshold
        attack_ratio = np.random.uniform(0.0, 0.8)
        n_attack     = int(max_nodes_per_graph * attack_ratio)
        n_benign     = max_nodes_per_graph - n_attack

        rows = []

        # Sample attack rows if any
        if n_attack > 0:
            attack_idx = np.random.choice(
                len(attack_df), size=n_attack, replace=True
            )
            rows.append(attack_df.iloc[attack_idx])

        # Sample benign rows
        if n_benign > 0:
            benign_idx = np.random.choice(
                len(benign_df), size=n_benign, replace=True
            )
            rows.append(benign_df.iloc[benign_idx])

        chunk = pd.concat(rows, ignore_index=True)

        if len(chunk) < min_nodes_per_graph:
            continue

        # Shuffle so attack nodes aren't always at the start
        # Forces model to find attacks anywhere in the graph
        chunk = chunk.sample(
            frac=1, random_state=i
        ).reset_index(drop=True)

        # Node features: 6 network flow statistics
        node_features = torch.tensor(
            chunk[feature_cols].values,
            dtype=torch.float32
        )

        n = len(chunk)

        # k-NN edges — identical for all graphs
        edge_index, edge_attr = build_knn_edges(
            n=n, k=5, max_nodes=max_nodes_per_graph
        )

        if not edge_index:
            continue

        edge_index = torch.tensor(
            edge_index, dtype=torch.long
        ).t().contiguous()

        edge_attr = torch.tensor(
            edge_attr, dtype=torch.float32
        )

        # Graph label: attack if >20% nodes are attack rows
        # 20% threshold makes the problem genuinely hard —
        # model must detect even sparse attack presence
        actual_ratio = n_attack / max_nodes_per_graph
        is_attack    = 1.0 if actual_ratio > 0.2 else 0.0

        # Most common MITRE stage in this graph
        mitre_id = int(chunk['mitre_id'].mode()[0])

        graph = Data(
            x            = node_features,
            edge_index   = edge_index,
            edge_attr    = edge_attr,
            y            = torch.tensor([is_attack],
                           dtype=torch.float32),
            mitre_id     = torch.tensor([mitre_id]),
            # Store actual ratio for fidelity scoring downstream
            attack_ratio = torch.tensor([actual_ratio],
                           dtype=torch.float32)
        )
        graphs.append(graph)

    attack_count = sum(1 for g in graphs if g.y.item() == 1.0)
    benign_count = sum(1 for g in graphs if g.y.item() == 0.0)
    print(f"  Built {len(graphs)} graphs")
    print(f"  Attack graphs (>20% attack nodes): {attack_count}")
    print(f"  Benign graphs (<=20% attack nodes): {benign_count}")
    return graphs


def summarize_graphs(graphs):
    """
    Prints dataset summary for verification before training.
    Check that attack/benign ratio is reasonable and
    node/edge counts match expectations.
    """
    attack    = sum(1 for g in graphs if g.y.item() == 1.0)
    benign    = sum(1 for g in graphs if g.y.item() == 0.0)
    avg_nodes = np.mean([g.num_nodes for g in graphs])
    avg_edges = np.mean([g.num_edges for g in graphs])

    print(f"\n=== GRAPH DATASET SUMMARY ===")
    print(f"  Total graphs    : {len(graphs)}")
    print(f"  Attack graphs   : {attack}")
    print(f"  Benign graphs   : {benign}")
    print(f"  Avg nodes/graph : {avg_nodes:.1f}")
    print(f"  Avg edges/graph : {avg_edges:.1f}")
    print(f"  Node feature dim: {graphs[0].x.shape[1]}")
    print(f"  Edge feature dim: {graphs[0].edge_attr.shape[1]}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Loading processed CICIDS data...")
    df = pd.read_parquet('data/processed/cicids_final.parquet')

    with open('data/processed/feature_cols_cicids.pkl', 'rb') as f:
        feature_cols = pickle.load(f)

    print(f"Loaded {len(df):,} rows, {len(feature_cols)} features")
    print(f"\nMITRE distribution:")
    print(df['mitre_stage'].value_counts())
    print()

    # ── SPLIT RAW ROWS 80/20 ──────────────────────────────────
    # Split rows BEFORE building graphs.
    # Train and val graphs are built from different rows.
    # Guarantees zero overlap — honest validation scores.
    print("Splitting raw rows 80/20...")

    train_df, val_df = train_test_split(
        df,
        test_size  = 0.2,
        random_state = 42,
        stratify   = df['mitre_stage']
    )

    print(f"  Train rows: {len(train_df):,}")
    print(f"  Val rows  : {len(val_df):,}")
    print(f"  Overlap   : "
          f"{len(set(train_df.index) & set(val_df.index))} rows\n")

    # ── BUILD TRAINING GRAPHS ─────────────────────────────────
    # 60,000 mixed graphs from training rows
    # Each graph: random mix of attack + benign nodes
    # Label: attack if >20% of nodes are attack rows
    print("Building TRAINING graphs (mixed attack+benign)...")
    train_graphs = build_mixed_graphs(
        train_df,
        feature_cols,
        max_nodes_per_graph = 50,
        min_nodes_per_graph = 5,
        n_graphs            = 60000
    )
    summarize_graphs(train_graphs)

    # ── BUILD VALIDATION GRAPHS ───────────────────────────────
    # 10,000 mixed graphs from validation rows
    # No augmentation — clean unmodified evaluation
    print("\nBuilding VALIDATION graphs (mixed attack+benign)...")
    val_graphs = build_mixed_graphs(
        val_df,
        feature_cols,
        max_nodes_per_graph = 50,
        min_nodes_per_graph = 5,
        n_graphs            = 10000
    )
    summarize_graphs(val_graphs)

    # ── SAVE ──────────────────────────────────────────────────
    print("\nSaving graphs...")
    with open('data/graphs/train_graphs.pkl', 'wb') as f:
        pickle.dump(train_graphs, f)

    with open('data/graphs/val_graphs.pkl', 'wb') as f:
        pickle.dump(val_graphs, f)

    print(f"Saved {len(train_graphs)} train graphs")
    print(f"Saved {len(val_graphs)} val graphs")
    print("Ready for GATv2 training.")