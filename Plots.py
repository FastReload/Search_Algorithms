import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import ast

# Define the log files
log_files = {
    "Forward Selection - Small Dataset": "forward_log_small.csv",
    "Forward Selection - Large Dataset": "forward_log_large.csv",
    "Backward Elimination - Small Dataset": "backward_log_small.csv",
    "Backward Elimination - Large Dataset": "backward_log_large.csv",
    "Forward Selection - Taiwan Bankruptcy Dataset": "forward_log_taiwan_final.csv",
}

for name, filepath in log_files.items():
    df = pd.read_csv(filepath)
    df["Feature_Indices"] = df["Feature_Indices"].apply(ast.literal_eval)
    df["Cumulative_Best_Accuracy"] = df["Accuracy"].cummax()

    # Plot 1: Accuracy vs Number of Features (replicating your sample image)
    plt.figure(figsize=(10, 6))
    plt.plot(df["Num_Features"], df["Accuracy"], color='orange', marker='o', linestyle='-')
    plt.title(f"Accuracy vs. Num Features - {name}", fontsize=14)
    plt.xlabel("Number of Features", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Top 10 Feature Sets by Accuracy with value labels
    top10_df = df.sort_values(by="Accuracy", ascending=False).head(10)
    accuracies = top10_df["Accuracy"][::-1]
    feature_labels = [str(f) for f in top10_df["Feature_Indices"][::-1]]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(10), accuracies, tick_label=feature_labels, color='orange')

    # Annotate each bar with accuracy
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{width:.2f}%", va='center', fontsize=11)

    plt.title(f"Top 10 Feature Sets by Accuracy\n({name})", fontsize=16)
    plt.xlabel("Accuracy (%)", fontsize=14)
    plt.ylabel("Feature Indices", fontsize=14)
    plt.xlim(0, 100)
    plt.grid(axis="x")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
