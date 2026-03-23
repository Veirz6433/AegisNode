from torch_geometric.datasets import EllipticBitcoinDataset

# The `root` parameter controls exactly where PyG saves raw + processed files.
# Using a subfolder like './data/elliptic' keeps everything organized and
# away from your root project directory.
dataset = EllipticBitcoinDataset(root='./data/elliptic')

# The dataset is a single large graph; access it via index 0
data = dataset[0]

print(f"Total number of nodes: {data.num_nodes}")
