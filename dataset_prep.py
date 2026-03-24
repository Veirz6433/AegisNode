import os
import torch
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.utils import subgraph
from torch_geometric.data import Data

# Load dataset
dataset = EllipticBitcoinDataset(root='./data/elliptic')
data = dataset[0]

total_nodes = data.num_nodes
print(f"Total number of nodes: {total_nodes}")

# Create output folder
os.makedirs("data_shards", exist_ok=True)

num_clients = 5
chunk_size = total_nodes // num_clients

for i in range(num_clients):
    start = i * chunk_size
    # Last client gets any remainder nodes too
    end = total_nodes if i == num_clients - 1 else (i + 1) * chunk_size

    # Define the node indices for this shard
    subset_node_indices = torch.arange(start, end)

    # Extract only the edges that exist within this chunk, re-index from 0
    edge_index, _ = subgraph(subset_node_indices, data.edge_index, relabel_nodes=True)

    # Slice features and labels for this chunk
    client_data = Data(
        x=data.x[start:end],
        y=data.y[start:end],
        edge_index=edge_index
    )

    torch.save(client_data, f"data_shards/client_{i}.pt")
    print(f"Client {i}: nodes={client_data.num_nodes}, edges={client_data.num_edges}")

print("\nAll 5 shards saved to data_shards/")
