# server.py — AegisNode with Prometheus Observability & Managed Experimentation
# ──────────────────────────────────────────────────────────────────────────────
import gc
import argparse
import numpy as np
import pandas as pd
import flwr as fl
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import (
    Parameters, FitRes, parameters_to_ndarrays, ndarrays_to_parameters, Scalar, Metrics
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from prometheus_client import (
    Gauge, Counter, start_http_server
)

# ══════════════════════════════════════════════════════════════════════════════
#  Prometheus Metric Definitions
# ══════════════════════════════════════════════════════════════════════════════

CLIENT_MAX_NORM = Gauge("aegis_client_max_norm", "Max L2 norm per client", ["client_id"])
CLIENT_ANOMALY_SCORE = Gauge("aegis_client_anomaly_score", "Anomaly score ratio", ["client_id"])
CLIENT_FLAGGED = Gauge("aegis_client_flagged_malicious", "1 if malicious, 0 otherwise", ["client_id"])
ACTIVE_CLIENTS = Gauge("aegis_active_clients_total", "Active clients in round")
MALICIOUS_DETECTIONS = Counter("aegis_malicious_detections_total", "Cumulative detections")
MALICIOUS_THIS_ROUND = Gauge("aegis_malicious_clients_this_round", "Malicious count this round")
TRAINING_ROUND = Gauge("aegis_training_round", "Current round number")
MODEL_ACCURACY = Gauge("aegis_model_accuracy", "Model accuracy")
MODEL_LOSS = Gauge("aegis_model_loss", "Model loss")
POISON_THRESHOLD = Gauge("aegis_poison_threshold", "Max norm threshold")

# ══════════════════════════════════════════════════════════════════════════════
#  Configuration & Argument Parsing
# ══════════════════════════════════════════════════════════════════════════════

POISON_NORM_THRESHOLD = 50.0   # Adjust based on your Elliptic dataset norm
METRICS_PORT = 8000
TOTAL_ROUNDS = 50              # Standard for your paper experiments

# ══════════════════════════════════════════════════════════════════════════════
#  ByzantineShield — Managed Strategy
# ══════════════════════════════════════════════════════════════════════════════

class ByzantineShield(FedAvg):
    def __init__(self, experiment_id: str, is_robust: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment_id = experiment_id
        self.is_robust = is_robust
        self.history = {"round": [], "accuracy": [], "loss": []}
        
        print(f"🛡️  ByzantineShield Init | Experiment: {experiment_id} | Robust Mode: {is_robust}")

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], 
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # --- Update Prometheus Meta-Metrics ---
        TRAINING_ROUND.set(server_round)
        POISON_THRESHOLD.set(POISON_NORM_THRESHOLD)
        ACTIVE_CLIENTS.set(len(results))

        if not results:
            return None, {}

        # --- Extract Weights and Run Observability ---
        client_weights = []
        malicious_count = 0
            
        print(f"\nRound {server_round} - Unpacking {len(results)} updates...")

        for client_proxy, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            max_norm = max(np.linalg.norm(w.flatten()) for w in weights)
            short_cid = client_proxy.cid[:8]
            anomaly_score = max_norm / POISON_NORM_THRESHOLD

            # Prometheus logging
            CLIENT_MAX_NORM.labels(client_id=short_cid).set(max_norm)
            CLIENT_ANOMALY_SCORE.labels(client_id=short_cid).set(anomaly_score)

            if max_norm > POISON_NORM_THRESHOLD:
                CLIENT_FLAGGED.labels(client_id=short_cid).set(1)
                MALICIOUS_DETECTIONS.inc()
                malicious_count += 1
            else:
                CLIENT_FLAGGED.labels(client_id=short_cid).set(0)

            client_weights.append(weights)

        MALICIOUS_THIS_ROUND.set(malicious_count)

        # --- Aggregation Decision ---
        if self.is_robust:
            # COORDINATE-WISE MEDIAN (AegisNode logic)
            print(f"🔢 Applying Coordinate-wise Median...")
            num_layers = len(client_weights[0])
            aggregated_ndarrays = []
            for layer_idx in range(num_layers):
                layer_stack = np.array([cw[layer_idx] for cw in client_weights])
                aggregated_ndarrays.append(np.median(layer_stack, axis=0))
        else:
            # STANDARD MEAN (FedAvg logic)
            print(f"⚖️  Applying Standard Federated Averaging...")
            aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)
            if aggregated_parameters is None: return None, {}
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

        # Cleanup
        gc.collect()
        
        # Aggregate metrics (accuracy/loss)
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return ndarrays_to_parameters(aggregated_ndarrays), metrics_aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if metrics and "accuracy" in metrics:
            acc = metrics["accuracy"]
            self.history["round"].append(server_round)
            self.history["accuracy"].append(acc)
            self.history["loss"].append(loss)
            
            # Prometheus Updates
            MODEL_ACCURACY.set(acc)
            MODEL_LOSS.set(loss)
            
            print(f"📉 Round {server_round} Results: Accuracy {acc:.4f} | Loss {loss:.4f}")

        # AUTO-SAVE at the end of the experiment
        if server_round == TOTAL_ROUNDS:
            df = pd.DataFrame(self.history)
            filename = f"results_{self.experiment_id}.csv"
            df.to_csv(filename, index=False)
            print(f"💾 Experiment Data saved to {filename}")

        return loss, metrics

# ══════════════════════════════════════════════════════════════════════════════
#  Server Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="test_run", help="Unique ID for CSV file")
    parser.add_argument("--robust", type=int, default=1, help="1 for AegisNode Median, 0 for FedAvg")
    args = parser.parse_args()

    # Start Prometheus
    print(f"📊 Starting Prometheus metrics on port {METRICS_PORT}...")
    start_http_server(METRICS_PORT)

    strategy = ByzantineShield(
        experiment_id=args.id,
        is_robust=bool(args.robust),
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4 if "baseline" in args.id else 5,
        min_evaluate_clients=4 if "baseline" in args.id else 5,
        min_available_clients=4 if "baseline" in args.id else 5,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print(f"\n🚀 Launching server: {args.id} (Robust={args.robust})\n")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=TOTAL_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()