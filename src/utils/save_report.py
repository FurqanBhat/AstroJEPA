import os
from src.inference.visualize_predictions import visualize_full_latent_map
import matplotlib.pyplot as plt


def save_anomaly_report(model, dataset, idx, score, save_path, device='cuda'):
    plt.figure(figsize=(24, 6))

    # Run your visualization function
    model.eval()
    visualize_full_latent_map(model, dataset, idx=idx, device=device)

    # Save
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved report: {save_path}")