import gc
import numpy as np
import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import Scalar


# ══════════════════════════════════════════════════════════════════════════════
#  ByzantineShield — Coordinate-wise Median Aggregation Strategy
# ══════════════════════════════════════════════════════════════════════════════

class ByzantineShield(FedAvg):
    """
    Custom Flower strategy that replaces FedAvg's mean aggregation
    with coordinate-wise median — robust against Byzantine clients.
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        print("\n" + "═" * 62)
        print(f"  🛡️  ByzantineShield — Round {server_round} AGGREGATION START")
        print(f"  📡  Clients responded : {len(results)}")
        print(f"  ❌  Failures          : {len(failures)}")
        print("═" * 62)

        if not results:
            print("  ⚠️  No results received. Skipping aggregation.")
            return None, {}

        # ── Step 1: Unpack + poison detection ────────────────────────────────
        print(f"  📦  Unpacking weight tensors from {len(results)} clients...\n")
        client_weights: List[List[np.ndarray]] = []

        for idx, (client_proxy, fit_res) in enumerate(results):
            weights  = parameters_to_ndarrays(fit_res.parameters)
            max_norm = max(np.linalg.norm(w.flatten()) for w in weights)

            # Flag any client whose weight norms are suspiciously large
            if max_norm > 50.0:
                poison_flag = "☠️  SUSPECTED POISON"
            else:
                poison_flag = "✅ Honest"

            print(f"      Client {idx} | cid: {client_proxy.cid[:8]}... | "
                  f"examples: {fit_res.num_examples:,} | "
                  f"max_norm: {max_norm:>10.2f} | {poison_flag}")

            client_weights.append(weights)

        # ── Step 2: Coordinate-wise Median ────────────────────────────────────
        print(f"\n  🔢  Computing coordinate-wise median across all clients...")
        num_layers     = len(client_weights[0])
        median_weights: List[np.ndarray] = []

        for layer_idx in range(num_layers):
            layer_stack  = np.array([
                client_weights[c][layer_idx] for c in range(len(client_weights))
            ])
            layer_median = np.median(layer_stack, axis=0)
            median_weights.append(layer_median)
            print(f"      Layer {layer_idx:02d} | shape: {str(layer_stack.shape[1:]):>20s} "
                  f"| median norm: {np.linalg.norm(layer_median.flatten()):>8.4f} ✔")

        print(f"\n  ✅  Median aggregation complete — {num_layers} layers processed.")

        # ── Step 3: Convert back to Flower Parameters ─────────────────────────
        parameters_aggregated = ndarrays_to_parameters(median_weights)

        # ── Step 4: RAM Health Bar ────────────────────────────────────────────
        print(f"  🧹  Purging raw client weight tensors from RAM...")
        del client_weights, median_weights, layer_stack, layer_median
        gc.collect()
        print(f"  💾  RAM cleanup complete.")
        print("═" * 62 + "\n")

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated


# ══════════════════════════════════════════════════════════════════════════════
#  Server Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║         AegisNode — Central Aggregation Server             ║")
    print("║         Strategy : ByzantineShield (Coord. Median)         ║")
    print("║         Rounds   : 30                                      ║")
    print("║         Address  : localhost:8080                          ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    strategy = ByzantineShield(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,        # need 3+ to outvote 1 rogue
        min_evaluate_clients=4,
        min_available_clients=4,  # wait for 3 before starting
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
