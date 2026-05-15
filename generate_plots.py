import pandas as pd
import matplotlib.pyplot as plt

def create_research_plot():
    plt.figure(figsize=(10, 6))
    
    files = {
        "Baseline (0% Rogue)": "results_baseline.csv",
        "FedAvg (20% Rogue)": "results_attacked.csv",
        "AegisNode (20% Rogue)": "results_aegisnode.csv"
    }
    
    colors = ['green', 'red', 'blue']
    styles = ['-', '--', '-']

    for (label, filename), color, style in zip(files.items(), colors, styles):
        try:
            df = pd.read_csv(filename)
            plt.plot(df['round'], df['accuracy'], label=label, color=color, linestyle=style, linewidth=2)
        except FileNotFoundError:
            print(f"⚠️ Warning: {filename} not found. Run the experiments first!")

    # Formatting for Academic Paper
    plt.title("AegisNode Resilience vs. Byzantine Model Poisoning", fontsize=14)
    plt.xlabel("Communication Rounds", fontsize=12)
    plt.ylabel("Global Model Accuracy", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(0, 1.0) # Accuracy is 0.0 to 1.0
    
    # Save as high-res PNG for the paper
    plt.savefig("aegisnode_performance_comparison.png", dpi=300)
    plt.show()
    print("✅ Research plot generated: aegisnode_performance_comparison.png")

if __name__ == "__main__":
    create_research_plot()