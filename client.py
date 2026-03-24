import os         #{only needed for local deployment}
# Force OpenMP and MKL to only use 4 threads BEFORE any heavy math libraries load         {only needed for local deployment}
os.environ["OMP_NUM_THREADS"] = "4"           #{only needed for local deployment}
os.environ["MKL_NUM_THREADS"] = "4"           #{only needed for local deployment}
os.environ["NUMEXPR_NUM_THREADS"] = "4"          # {only needed for local deployment}

import gc
import argparse
import numpy as np
import torch
torch.set_num_threads(4) #{only needed for local deployment}
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
import flwr as fl
from flwr.common import NDArrays
from typing import Tuple, Dict


# ── 1. AegisSAGE Model ────────────────────────────────────────────────────────
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
        return x


# ── 2. Local DP Training Function ─────────────────────────────────────────────
def train_local_model(model, train_loader, optimizer, epochs=1):
    import math
    MAX_GRAD_NORM    = 1.0
    NOISE_MULTIPLIER = 1.0
    DELTA            = 1e-5

    model.train()
    final_loss  = 0.0
    total_steps = 0

    for epoch in range(epochs):
        epoch_loss  = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            out      = model(batch.x, batch.edge_index)
            seed_out = out[:batch.batch_size]
            seed_y   = batch.y[:batch.batch_size]

            mask = (seed_y == 1) | (seed_y == 2)
            if mask.sum() == 0:
                del batch, out, seed_out, seed_y, mask
                gc.collect()
                continue

            remapped_y = seed_y[mask] - 1
            loss = F.cross_entropy(seed_out[mask], remapped_y.long())

            loss.backward()

            # DP Step 1: Clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            # DP Step 2: Inject Gaussian noise
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.randn_like(param.grad) * NOISE_MULTIPLIER * MAX_GRAD_NORM
                        param.grad += noise

            optimizer.step()

            epoch_loss  += loss.item()
            num_batches += 1
            total_steps += 1

            del batch, out, seed_out, seed_y, mask, remapped_y, loss
            gc.collect()

        avg_loss   = epoch_loss / max(num_batches, 1)
        final_loss = avg_loss
        print(f"    Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

    epsilon = (1.0 / NOISE_MULTIPLIER**2) * math.sqrt(
        2 * total_steps * math.log(1.0 / DELTA)
    )
    print(f"    🔒 ε ≈ {epsilon:.4f}  (δ = {DELTA})")

    return final_loss, epsilon


# ── 3. Flower NumPyClient Wrapper ─────────────────────────────────────────────
class AegisFlowerClient(fl.client.NumPyClient):

    def __init__(self, client_id: int):
        self.client_id = client_id

        self.data = torch.load(
            f"data_shards/client_{client_id}.pt", weights_only=False
        )
        self.loader = NeighborLoader(
            self.data,
            num_neighbors=[10, 10],
            batch_size=64,
            input_nodes=None,
            shuffle=True,
        )
        self.model     = AegisSAGE(in_channels=165, hidden_channels=64, out_channels=2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        print(f"  [Client {client_id}] ✅ Initialized — "
              f"nodes: {self.data.num_nodes}, edges: {self.data.num_edges}")

    def get_parameters(self, config) -> NDArrays:
        print(f"  [Client {self.client_id}] 📤 Sending parameters to server...")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays):
        print(f"  [Client {self.client_id}] 📥 Receiving global parameters...")
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(self.model.state_dict().keys(), parameters)
        }
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config) -> Tuple[NDArrays, int, Dict]:
        print(f"\n  [Client {self.client_id}] 🏋️  fit() called...")
        self.set_parameters(parameters)

        loss, epsilon = train_local_model(
            self.model, self.loader, self.optimizer, epochs=1
        )

        print(f"  [Client {self.client_id}] ✅ fit() done — "
              f"loss: {loss:.4f} | ε: {epsilon:.4f}")

        return self.get_parameters(config={}), self.data.num_nodes, {
            "loss":    float(loss),
            "epsilon": float(epsilon),
        }

    def evaluate(self, parameters: NDArrays, config) -> Tuple[float, int, Dict]:
        print(f"  [Client {self.client_id}] 📊 evaluate() called...")
        self.set_parameters(parameters)

        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for batch in self.loader:
                out      = self.model(batch.x, batch.edge_index)
                seed_out = out[:batch.batch_size]
                seed_y   = batch.y[:batch.batch_size]

                mask = (seed_y == 1) | (seed_y == 2)
                if mask.sum() == 0:
                    continue

                remapped_y = seed_y[mask] - 1
                loss       = F.cross_entropy(seed_out[mask], remapped_y.long())
                total_loss += loss.item()

                preds    = seed_out[mask].argmax(dim=1)
                correct += (preds == remapped_y).sum().item()
                total   += mask.sum().item()

                del batch, out, seed_out, seed_y, mask, remapped_y
                gc.collect()

        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)

        print(f"  [Client {self.client_id}] 📈 Accuracy: {accuracy:.4f} | "
              f"Loss: {avg_loss:.4f}")

        return float(avg_loss), self.data.num_nodes, {"accuracy": float(accuracy)}


# ── 4. Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client-id", type=int, required=True,
        help="Client ID 0–4, determines which shard to load"
    )
    args = parser.parse_args()

    print(f"\n🚀 Starting AegisNode Flower Client {args.client_id}...")

    fl.client.start_client(
        server_address="localhost:8080",
        client=AegisFlowerClient(client_id=args.client_id).to_client(),
    )
