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
    with coordinate-wise median — robust against Byzantine clients
    (poisoned gradients, adversarial updates, or corrupted shards).
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        print("\n" + "═" * 60)
        print(f"  🛡️  ByzantineShield — Round {server_round} AGGREGATION START")
        print(f"  📡  Clients responded : {len(results)}")
        print(f"  ❌  Failures          : {len(failures)}")
        print("═" * 60)

        if not results:
            print("  ⚠️  No results received. Skipping aggregation.")
            return None, {}

        # ── Step 1: Unpack client weights from Flower's Parameters format ─────
        print(f"  📦  Unpacking weight tensors from {len(results)} clients...")
        client_weights: List[List[np.ndarray]] = []

        for client_proxy, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            client_weights.append(weights)
            print(f"      ✔ Client unpacked — "
                  f"{fit_res.num_examples} examples, "
                  f"{len(weights)} weight tensors")

        # ── Step 2: Coordinate-wise Median ────────────────────────────────────
        print(f"\n  🔢  Computing coordinate-wise median across all clients...")
        num_layers = len(client_weights[0])
        median_weights: List[np.ndarray] = []

        for layer_idx in range(num_layers):
            # Stack this layer's weights from all clients → shape: [num_clients, *layer_shape]
            layer_stack = np.array([
                client_weights[c][layer_idx] for c in range(len(client_weights))
            ])
            # Median across axis=0 (the client axis) — Byzantine-robust
            layer_median = np.median(layer_stack, axis=0)
            median_weights.append(layer_median)
            print(f"      Layer {layer_idx:02d} | shape: {layer_stack.shape[1:]} "
                  f"| median computed ✔")

        print(f"\n  ✅  Median aggregation complete — {num_layers} layers processed.")

        # ── Step 3: Convert back to Flower Parameters format ──────────────────
        parameters_aggregated = ndarrays_to_parameters(median_weights)

        # ── Step 4: RAM Health Bar — purge raw weights immediately ────────────
        print(f"  🧹  Purging raw client weight tensors from RAM...")
        del client_weights, median_weights, layer_stack, layer_median
        gc.collect()
        print(f"  💾  RAM cleanup complete.")
        print("═" * 60 + "\n")

        # Aggregate metrics if a function was provided (pass-through to FedAvg)
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated


# ══════════════════════════════════════════════════════════════════════════════
#  Server Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         AegisNode — Central Aggregation Server           ║")
    print("║         Strategy : ByzantineShield (Coord. Median)       ║")
    print("║         Rounds   : 30                                    ║")
    print("║         Address  : localhost:8080                        ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    strategy = ByzantineShield(
        fraction_fit=1.0,           # use 100% of available clients each round
        fraction_evaluate=1.0,      # evaluate on 100% of clients
        min_fit_clients=2,          # need at least 2 clients to start a round
        min_evaluate_clients=2,
        min_available_clients=2,    # wait until at least 2 clients connect
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
