import pandas as pd

def calculate_final_metrics():
    print("="*50)
    print("  FINAL METRICS (Last 50 Rounds)  ")
    print("="*50)
    
    files = {
        "Baseline (0% Rogue)": "results_baseline.csv",
        "FedAvg (20% Rogue)": "results_attacked.csv",
        "AegisNode (20% Rogue)": "results_aegisnode.csv"
    }
    
    for label, filename in files.items():
        try:
            df = pd.read_csv(filename)
            # Get the accuracy of the last 10 rounds
            last_10 = df['accuracy'].tail(50)
            
            # Calculate Mean and Standard Deviation
            mean_acc = last_10.mean() * 100
            std_dev = last_10.std() * 100
            
            print(f"{label:<25} | {mean_acc:.2f}% ± {std_dev:.2f}%")
        except FileNotFoundError:
            print(f"{label:<25} | File not found!")

if __name__ == "__main__":
    calculate_final_metrics()