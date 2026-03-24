# server.py — AegisNode with Full Prometheus Observability
# ─────────────────────────────────────────────────────────
# pip install prometheus_client flwr numpy

import gc
import threading
import numpy as np
import flwr as fl

from flwr.common import (
    Parameters, FitRes,
    parameters_to_ndarrays, ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import Scalar

from prometheus_client import (
    Gauge, Counter, start_http_server, REGISTRY
)

# ══════════════════════════════════════════════════════════════════════════════
#  Prometheus Metric Definitions
#  Naming convention: aegis_<subsystem>_<metric>
# ══════════════════════════════════════════════════════════════════════════════

# Max L2 norm of weight tensors for each client.
# High values indicate potential model poisoning / Byzantine behavior.
# Label: client_id (truncated CID for cardinality safety)
CLIENT_MAX_NORM = Gauge(
    "aegis_client_max_norm",
    "Maximum L2 norm of weight tensors submitted by a client in a round",
    labelnames=["client_id"],
)

# Anomaly score per client: normalized deviation from median norm.
# Score = max_norm / POISON_THRESHOLD; score > 1.0 means suspected poison.
CLIENT_ANOMALY_SCORE = Gauge(
    "aegis_client_anomaly_score",
    "Anomaly score for each client (ratio of max_norm to poison threshold)",
    labelnames=["client_id"],
)

# Binary flag: 1 if client was flagged as malicious, 0 otherwise.
# Allows per-client time-series alert tracking in Grafana.
CLIENT_FLAGGED = Gauge(
    "aegis_client_flagged_malicious",
    "1 if client was flagged as malicious this round, else 0",
    labelnames=["client_id"],
)

# ── Aggregate / System Metrics ────────────────────────────────────────────────

# Total clients that responded this round (resets each round).
ACTIVE_CLIENTS = Gauge(
    "aegis_active_clients_total",
    "Number of clients that responded in the current training round",
)

# Running count of unique malicious detection events across all rounds.
# Use Counter (never decrements) so Grafana rate() works correctly.
MALICIOUS_DETECTIONS = Counter(
    "aegis_malicious_detections_total",
    "Cumulative number of clients flagged as malicious across all rounds",
)

# Current count of flagged clients in the LATEST round (resets each round).
MALICIOUS_THIS_ROUND = Gauge(
    "aegis_malicious_clients_this_round",
    "Number of malicious clients detected in the current round",
)

# Current training round index (1-based).
TRAINING_ROUND = Gauge(
    "aegis_training_round",
    "Current federated learning round number",
)

# Aggregated model accuracy reported by clients (weighted average from metrics).
MODEL_ACCURACY = Gauge(
    "aegis_model_accuracy",
    "Aggregated model accuracy across all honest clients",
)

# Aggregated model loss reported by clients.
MODEL_LOSS = Gauge(
    "aegis_model_loss",
    "Aggregated model loss across all honest clients",
)

# Threshold used for poison detection — exported so Grafana can draw a line.
POISON_THRESHOLD = Gauge(
    "aegis_poison_threshold",
    "The max_norm threshold above which a client is flagged as malicious",
)

# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

POISON_NORM_THRESHOLD = 50.0   # L2 norm threshold for Byzantine detection
METRICS_PORT          = 8000   # Prometheus scrape endpoint port


# ══════════════════════════════════════════════════════════════════════════════
#  ByzantineShield — Coord-wise Median + Prometheus Observability
# ══════════════════════════════════════════════════════════════════════════════

class ByzantineShield(FedAvg):
    """
    Robust aggregation strategy with real-time Prometheus telemetry.
    Replaces FedAvg mean with coordinate-wise median; flags Byzantine clients.
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

        # ── Emit static threshold so Grafana can render a reference line ──────
        POISON_THRESHOLD.set(POISON_NORM_THRESHOLD)

        # ── Update round counter ───────────────────────────────────────────────
        TRAINING_ROUND.set(server_round)

        if not results:
            print("  ⚠️  No results. Skipping aggregation.")
            ACTIVE_CLIENTS.set(0)
            return None, {}

        # ── Step 1: Unpack weights + per-client metrics ────────────────────────
        ACTIVE_CLIENTS.set(len(results))
        client_weights: List[List[np.ndarray]] = []
        malicious_count = 0

        print(f"\n  📦  Unpacking {len(results)} client updates...\n")

        for idx, (client_proxy, fit_res) in enumerate(results):
            weights  = parameters_to_ndarrays(fit_res.parameters)
            max_norm = max(np.linalg.norm(w.flatten()) for w in weights)

            # Truncate CID to 8 chars for safe Prometheus label cardinality
            short_cid = client_proxy.cid[:8]

            # Anomaly score: 1.0 = exactly at threshold, >1.0 = dangerous
            anomaly_score = max_norm / POISON_NORM_THRESHOLD

            # ── Push per-client metrics to Prometheus ──────────────────────
            CLIENT_MAX_NORM.labels(client_id=short_cid).set(max_norm)
            CLIENT_ANOMALY_SCORE.labels(client_id=short_cid).set(anomaly_score)

            if max_norm > POISON_NORM_THRESHOLD:
                poison_flag = "☠️  SUSPECTED POISON"
                CLIENT_FLAGGED.labels(client_id=short_cid).set(1)
                MALICIOUS_DETECTIONS.inc()   # Persistent counter
                malicious_count += 1
            else:
                poison_flag = "✅ Honest"
                CLIENT_FLAGGED.labels(client_id=short_cid).set(0)

            print(f"      Client {idx} | cid: {short_cid}... | "
                  f"examples: {fit_res.num_examples:,} | "
                  f"max_norm: {max_norm:>10.2f} | "
                  f"anomaly: {anomaly_score:.3f} | {poison_flag}")

            client_weights.append(weights)

        # ── Update round-level malicious count ─────────────────────────────
        MALICIOUS_THIS_ROUND.set(malicious_count)

        # ── Step 2: Coordinate-wise Median ────────────────────────────────────
        print(f"\n  🔢  Computing coordinate-wise median...")
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

        parameters_aggregated = ndarrays_to_parameters(median_weights)

        # ── Step 3: Aggregate model accuracy/loss from client metrics ─────────
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        # Push aggregated accuracy/loss if clients report them
        if "accuracy" in metrics_aggregated:
            MODEL_ACCURACY.set(metrics_aggregated["accuracy"])
        if "loss" in metrics_aggregated:
            MODEL_LOSS.set(metrics_aggregated["loss"])

        # ── RAM Cleanup ────────────────────────────────────────────────────────
        del client_weights, median_weights, layer_stack, layer_median
        gc.collect()

        print(f"  💾  RAM cleanup complete.")
        print("═" * 62 + "\n")

        return parameters_aggregated, metrics_aggregated


# ══════════════════════════════════════════════════════════════════════════════
#  Server Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Start Prometheus HTTP metrics server in background thread ─────────────
    print(f"  📊  Starting Prometheus metrics endpoint on port {METRICS_PORT}...")
    start_http_server(METRICS_PORT)
    print(f"  ✅  Metrics available at http://localhost:{METRICS_PORT}/metrics\n")

    print("╔════════════════════════════════════════════════════════════╗")
    print("║         AegisNode — Central Aggregation Server             ║")
    print("║         Strategy : ByzantineShield (Coord. Median)         ║")
    print("║         Rounds   : 30                                      ║")
    print("║         Metrics  : http://localhost:8000/metrics           ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    strategy = ByzantineShield(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
