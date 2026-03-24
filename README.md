# AegisNode 🛡️
**A Byzantine-Robust Federated Learning Framework for Privacy-Preserving Graph Intelligence**

Traditional Graph Neural Network (GNN) training paradigms struggle with strict data privacy silos, high communication costs, and a critical vulnerability to "Byzantine" faults—where malicious or failing nodes upload "poisoned" gradients to sabotage the global model. 

**AegisNode** is a decentralized, edge-optimized architecture designed to collaboratively train GNNs across isolated data silos (e.g., banks or hospitals) while mathematically guaranteeing structural privacy and neutralizing adversarial attacks in real-time.

---

## 🚀 Core Architecture & Features

* **The Brain (Graph Intelligence):** Utilizes **GraphSAGE** via *PyTorch Geometric (PyG)*. Implements fixed-size neighbor sampling (`NeighborLoader`) to prevent RAM-crashing "neighbor explosions," making it viable for memory-constrained edge devices (tested on strictly 16GB RAM limits).
* **The Infrastructure (Decentralized Orchestration):** Powered by the **Flower (`flwr`)** framework, enabling asynchronous gRPC communication between the central server and geographically distributed clients.
* **The Vault (Local Differential Privacy):** Integrates **Opacus** to inject calibrated Gaussian noise and apply strict $L_2$ gradient clipping at the local client level. This mathematical cloak protects against membership inference and model inversion attacks.
* **The Shield (Byzantine Fault Tolerance):** Replaces standard FedAvg with a custom **Coordinate-wise Median** aggregation strategy. This guarantees the global model's integrity even if up to 49% of the participating network nodes are compromised or actively attempting to inject poisoned gradients.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, PyTorch Geometric (PyG)
* **Federation:** Flower (flwr)
* **Privacy & Security:** Opacus (Differential Privacy), Custom Median Aggregators
* **Dataset:** Elliptic Bitcoin Dataset (Anti-Money Laundering classification)

---

## ⚙️ Installation & Setup

**1. Clone the Repository & Environment Setup**
It is highly recommended to use a virtual environment to prevent dependency conflicts.
```bash
git clone [https://github.com/yourusername/AegisNode.git](https://github.com/yourusername/AegisNode.git)
cd AegisNode
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio torch-geometric flwr opacus