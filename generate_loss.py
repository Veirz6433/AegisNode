import pandas as pd
import matplotlib.pyplot as plt

def create_loss_plot():
    plt.figure(figsize=(10, 6))
    
    # Define the data sources and their styling
    files = {
        "Baseline (0% Rogue)": "results_baseline.csv",
        "FedAvg (20% Rogue)": "results_attacked.csv",
        "AegisNode (20% Rogue)": "results_aegisnode.csv"
    }
    
    colors = ['green', 'red', 'blue']
    styles = ['-', '--', '-']

    # Read and plot each CSV
    for (label, filename), color, style in zip(files.items(), colors, styles):
        try:
            df = pd.read_csv(filename)
            # Plotting 'round' on X-axis and 'loss' on Y-axis
            plt.plot(df['round'], df['loss'], label=label, color=color, linestyle=style, linewidth=2)
        except FileNotFoundError:
            print(f"⚠️ Warning: {filename} not found. Ensure the experiment has run.")
        except KeyError:
            print(f"⚠️ Warning: 'loss' column not found in {filename}.")

    # Formatting for Academic Paper
    plt.title("Global Model Loss Under Byzantine Attack", fontsize=14)
    plt.xlabel("Communication Rounds", fontsize=12)
    plt.ylabel("Global Model Loss", fontsize=12)
    plt.legend(loc="upper right") # Moved to upper right since loss typically drops to the bottom left
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Optional: Use a logarithmic scale if the red line explodes too violently
    # plt.yscale('log') 
    
    # Save as high-res PNG for the paper
    plt.savefig("aegisnode_loss_comparison.png", dpi=300)
    plt.show()
    print("✅ Loss plot generated: aegisnode_loss_comparison.png")

if __name__ == "__main__":
    create_loss_plot()