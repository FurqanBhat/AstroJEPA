from src.inference.load_model import load_jepa_checkpoint
from src.inference.compute_anomaly_scores import compute_jepa_anomaly_scores
from src.data.data_processing import get_test_subset
from src.inference.visualize_predictions import visualize_full_latent_map
from src.utils.save_report import save_anomaly_report
from src.inference.extract_latents import extract_latents
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


model_paths = ['checkpoints/300cosmic_jepa_latest.pth', 'checkpoints/100jepa.pth', 'checkpoints/100lejepa.pth' ]

dataset = get_test_subset()

model, config = load_jepa_checkpoint(model_paths[0])
visualize_full_latent_map(model,dataset)    

# anomaly_df = compute_jepa_anomaly_scores(
#     model=model,
#     dataset=dataset,
#     batch_size=16, 
#     save_path='outputs/galaxy_anomaly_scores.csv'
#     )



# top_k = anomaly_df.sort_values("anomaly_score", ascending=False).head(5)


