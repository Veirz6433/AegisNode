import gc
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from opacus import PrivacyEngine


# ── 1. Load a single shard ────────────────────────────────────────────────────
data = torch.load("data_shards/client_0.pt", weights_only=False)
print(f"Shard loaded → nodes: {data.num_nodes}, edges: {data.num_edges}")
print(f"Feature shape: {data.x.shape}  |  Label shape: {data.y.shape}")


# ── 2. NeighborLoader (RAM Saver) ─────────────────────────────────────────────
loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    batch_size=64,
    input_nodes=None,
    shuffle=True,
)


# ── 3. AegisSAGE Model ────────────────────────────────────────────────────────
class AegisSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # raw logits — CrossEntropyLoss handles softmax


# ── 4. Local DP Training Function ─────────────────────────────────────────────
def train_local_model(model, train_loader, optimizer, epochs=3):
    """
    DP-SGD compatible with PyG NeighborLoader.
    Uses manual per-sample gradient clipping + Gaussian noise
    instead of make_private() (which breaks on PyG's Data objects).
    """
    MAX_GRAD_NORM  = 1.0
    NOISE_MULTIPLIER = 1.0
    DELTA          = 1e-5

    model.train()
    final_loss = 0.0
    total_steps = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index)

            seed_out = out[:batch.batch_size]
            seed_y   = batch.y[:batch.batch_size]

            mask = (seed_y == 1) | (seed_y == 2)
            if mask.sum() == 0:
                del batch, out, seed_out, seed_y, mask
                gc.collect()
                continue

            remapped_y = seed_y[mask] - 1
            loss = F.cross_entropy(seed_out[mask], remapped_y.long())

            # loss = F.cross_entropy(seed_out[mask], seed_y[mask].long())
            loss.backward()

            # ── DP Step 1: Gradient Clipping ──────────────────────────────────
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            # ── DP Step 2: Gaussian Noise Injection ───────────────────────────
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.randn_like(param.grad) * NOISE_MULTIPLIER * MAX_GRAD_NORM
                        param.grad += noise

            optimizer.step()

            epoch_loss  += loss.item()
            num_batches += 1
            total_steps += 1

            # ── RAM Health Bar ─────────────────────────────────────────────────
            del batch, out, seed_out, seed_y, mask, loss
            gc.collect()

        avg_loss   = epoch_loss / max(num_batches, 1)
        final_loss = avg_loss
        print(f"  Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

    # ── Privacy Budget Estimate (RDP Accountant approximation) ────────────────
    # ε ≈ noise_multiplier^-2 * sqrt(2 * steps * ln(1/δ))
    import math
    epsilon = (1.0 / NOISE_MULTIPLIER**2) * math.sqrt(
        2 * total_steps * math.log(1.0 / DELTA)
    )
    print(f"\n🔒 LDP Privacy Budget spent: ε ≈ {epsilon:.4f}  (δ = {DELTA})")
    print(f"   Interpretation: {'⚠ Weak privacy' if epsilon > 10 else '✅ Strong privacy'}")

    return final_loss, epsilon


# ── 5. Sanity Check — single forward pass (no training) ───────────────────────
model = AegisSAGE(in_channels=165, hidden_channels=64, out_channels=2)
print(f"\nModel architecture:\n{model}\n")

batch = next(iter(loader))
print(f"Batch → nodes: {batch.num_nodes}, edges: {batch.num_edges}")

model.eval()
with torch.no_grad():
    out = model(batch.x, batch.edge_index)

print(f"Output tensor shape: {out.shape}")
print("\n✅ Engine check passed. AegisSAGE is ready.")


# ── 6. Run a real DP training session ─────────────────────────────────────────
print("\n--- Starting Local DP Training ---")
model = AegisSAGE(in_channels=165, hidden_channels=64, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

final_loss, epsilon = train_local_model(model, loader, optimizer, epochs=3)
print(f"\nTraining complete → Loss: {final_loss:.4f}  |  ε: {epsilon:.4f}")
