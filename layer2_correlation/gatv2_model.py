import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader
import pickle
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# WHY GATv2:
# Standard GAT computes attention weights based only on the
# query node — it cannot differentiate neighbors well.
# GATv2 fixes this by computing attention using BOTH nodes
# in each edge — much better at identifying which alert
# relationships are actually part of an attack chain.
#
# VRAM-optimized settings for RTX 4050 6GB:
# hidden_dim = 128  — good capacity within VRAM budget
# heads      = 4    — 4 parallel attention perspectives
# batch_size = 32   — safe with ~427 edges per graph
# edge_dim   = 1    — single distance feature per edge
#                     no class signal in edges
# ─────────────────────────────────────────────────────────────

class AttackGraphGATv2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128,
                 heads=4, edge_dim=1, dropout=0.3):
        super().__init__()

        # ── 4 GATv2 Convolutional layers ─────────────────────
        # Layer 1: input_dim (6) → hidden_dim*heads (512)
        self.conv1 = GATv2Conv(
            input_dim, hidden_dim,
            heads=heads, edge_dim=edge_dim, concat=True
        )
        # Layer 2: 512 → 512
        self.conv2 = GATv2Conv(
            hidden_dim * heads, hidden_dim,
            heads=heads, edge_dim=edge_dim, concat=True
        )
        # Layer 3: 512 → 512 (receives skip from layer 1)
        self.conv3 = GATv2Conv(
            hidden_dim * heads, hidden_dim,
            heads=heads, edge_dim=edge_dim, concat=True
        )
        # Layer 4: 512 → 128 (single head, reduces dimension)
        self.conv4 = GATv2Conv(
            hidden_dim * heads, hidden_dim,
            heads=1, edge_dim=edge_dim, concat=False
        )

        # Skip connection from layer 1 to layer 3
        # Prevents vanishing gradients in deep networks
        # Lets layer 3 combine shallow and deep features
        self.skip_proj = torch.nn.Linear(
            hidden_dim * heads, hidden_dim * heads
        )

        # Batch normalization after each conv layer
        # Keeps activations in consistent range
        # Makes training stable and faster to converge
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * heads)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim * heads)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim * heads)

        self.dropout_p = dropout

        # Final classifier
        # Input : 128-dim graph embedding
        # Output: single attack probability (0 to 1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr, batch):

        # Layer 1 — initial feature transformation
        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = self.bn1(x1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout_p,
                        training=self.training)

        # Layer 2 — deeper pattern extraction
        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = self.bn2(x2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=self.dropout_p,
                        training=self.training)

        # Layer 3 — with skip connection from layer 1
        # Adds layer 1 output to prevent information loss
        x3 = self.conv3(x2, edge_index, edge_attr)
        x3 = self.bn3(x3)
        x3 = F.elu(x3 + self.skip_proj(x1))
        x3 = F.dropout(x3, p=self.dropout_p,
                        training=self.training)

        # Layer 4 — single head, compresses to hidden_dim
        x4 = self.conv4(x3, edge_index, edge_attr)

        # Global mean pooling
        # Averages all node embeddings → one 128-dim vector
        # Converts variable-size graphs to fixed-size vectors
        graph_emb = global_mean_pool(x4, batch)

        return self.classifier(graph_emb)


def train_epoch(model, loader, optimizer, device, class_weights):
    """
    Runs one complete pass through all training graphs.

    Weighted BCE loss:
    Penalizes missing a real attack more than a false positive.
    In banking security, missing an attack is catastrophic.
    False positives are annoying but manageable.
    """
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(
            batch.x, batch.edge_index,
            batch.edge_attr, batch.batch
        )

        # Higher weight on attack errors than benign errors
        weights = torch.where(
            batch.y == 1,
            torch.tensor(class_weights[1], device=device),
            torch.tensor(class_weights[0], device=device)
        )
        loss = F.binary_cross_entropy(
            out.squeeze(), batch.y, weight=weights
        )
        loss.backward()

        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    """
    Evaluates model without updating weights.

    4 metrics:
    Accuracy  = (TP+TN)/total
    Precision = TP/(TP+FP) — quality of attack predictions
    Recall    = TP/(TP+FN) — coverage of real attacks
    F1        = 2*P*R/(P+R) — primary metric, balances both
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out   = model(
                batch.x, batch.edge_index,
                batch.edge_attr, batch.batch
            )
            preds = (out.squeeze() > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy  = (all_preds == all_labels).mean()
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return accuracy, precision, recall, f1


if __name__ == "__main__":

    # Fixed seeds — same results every run
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # ── Load pre-split graphs ─────────────────────────────────
    # Train and val built from non-overlapping rows
    # Zero data leakage guaranteed
    print("Loading graphs...")

    with open('data/graphs/train_graphs.pkl', 'rb') as f:
        train_graphs = pickle.load(f)

    with open('data/graphs/val_graphs.pkl', 'rb') as f:
        val_graphs = pickle.load(f)

    print(f"Train graphs: {len(train_graphs)}")
    print(f"Val graphs  : {len(val_graphs)}")

    # ── Class weights ─────────────────────────────────────────
    # Calculated from training set only
    n_benign = sum(1 for g in train_graphs if g.y.item() == 0.0)
    n_attack = sum(1 for g in train_graphs if g.y.item() == 1.0)
    total    = n_benign + n_attack
    w_benign = total / (2 * n_benign)
    w_attack = total / (2 * n_attack)
    class_weights = [w_benign, w_attack]
    print(f"Class weights — Benign: {w_benign:.3f} "
          f"| Attack: {w_attack:.3f}")

    # ── DataLoaders ───────────────────────────────────────────
    # batch_size=32: safe for 6GB VRAM with this architecture
    train_loader = DataLoader(
        train_graphs, batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        val_graphs, batch_size=32, shuffle=False
    )

    # ── Device ────────────────────────────────────────────────
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"\nTraining on: {device}")

    # ── Model ─────────────────────────────────────────────────
    # edge_dim=1: single distance feature, no class signal
    input_dim = train_graphs[0].x.shape[1]
    model = AttackGraphGATv2(
        input_dim  = input_dim,
        hidden_dim = 128,
        heads      = 4,
        edge_dim   = 1,
        dropout    = 0.3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    if device.type == 'cuda':
        vram = torch.cuda.memory_allocated(0) / 1024**3
        print(f"VRAM after model load: {vram:.2f} GB")

    # ── Optimizer ─────────────────────────────────────────────
    # Adam with L2 regularization
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-4
    )

    # Halve lr when F1 plateaus for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # ── Training loop ─────────────────────────────────────────
    print("\nStarting training...")
    print("=" * 70)

    best_f1          = 0.0
    best_epoch       = 0
    train_losses     = []
    val_f1s          = []
    patience_counter = 0
    early_stop       = 15

    for epoch in range(1, 101):

        loss = train_epoch(
            model, train_loader, optimizer,
            device, class_weights
        )
        acc, prec, rec, f1 = evaluate(
            model, val_loader, device
        )

        train_losses.append(loss)
        val_f1s.append(f1)
        scheduler.step(f1)

        if epoch % 5 == 0 or epoch == 1:
            if device.type == 'cuda':
                vram = torch.cuda.memory_allocated(0) / 1024**3
                print(
                    f"Epoch {epoch:03d} | "
                    f"Loss: {loss:.4f} | "
                    f"Acc: {acc:.4f} | "
                    f"Prec: {prec:.4f} | "
                    f"Rec: {rec:.4f} | "
                    f"F1: {f1:.4f} | "
                    f"VRAM: {vram:.2f}GB"
                )
            else:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Loss: {loss:.4f} | "
                    f"Acc: {acc:.4f} | "
                    f"F1: {f1:.4f}"
                )

        if f1 > best_f1:
            best_f1    = f1
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                'models/gatv2/best_model.pt'
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"No improvement for {early_stop} epochs")
            break

    print("=" * 70)
    print(f"\nTraining complete!")
    print(f"Best F1   : {best_f1:.4f}")
    print(f"Best epoch: {best_epoch}")

    # ── Save training curves ──────────────────────────────────
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, color='steelblue')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_f1s, color='darkorange')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('models/gatv2/training_curves.png', dpi=150)
    print("Training curves saved to models/gatv2/training_curves.png")

    # ── Final evaluation on best saved model ─────────────────
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(
        torch.load('models/gatv2/best_model.pt',
                   weights_only=True)
    )
    acc, prec, rec, f1 = evaluate(model, val_loader, device)

    print(f"\nFinal Results on Validation Set:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")